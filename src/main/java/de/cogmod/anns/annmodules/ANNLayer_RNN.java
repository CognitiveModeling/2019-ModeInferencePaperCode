package de.cogmod.anns.annmodules;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;

import de.cogmod.anns.errorlearners.AdamLearner;
import de.cogmod.anns.errorlearners.ErrorBasedLearner;
import de.cogmod.anns.errorlearners.ErrorBasedLearner.WEIGHT_UPDATE;
import de.cogmod.anns.errorlearners.GradientDescentLearner;
import de.cogmod.utilities.ActivationFunctions;
import de.cogmod.utilities.ActivationFunctions.ACT_FUNCT;
import de.cogmod.utilities.LearningListener;
import de.cogmod.utilities.Tools;

/**
 * Simple implementation of a recurrent neural network with 
 * hyperbolic tangent neurons and trainable biases.
 * 
 * @author Sebastian Otte & Martin V. Butz
 */
public class ANNLayer_RNN implements Serializable,ANNLayer{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public final static double BIAS = 1.0;

	private int layersnum;
	private int inputSize;
	private int outputSize;
	private int outputlayer;

	private int numWeights;
	private int[] layerSizes;
	private double[][][] net; // incoming activation of each neuron [layer][index][time]
	private double[][][] act; // activation of each neuron after neural function [layer][index][time]
	private double[][][] bwbuffer; // [layer][index][time]
	private double[][][] delta; // [layer][index][time]

	private boolean[] usebias; // index 0 = input-to-first-hidden-layer
	private final ACT_FUNCT[] actFuncts;  // index 0 = activation function in first hidden-layer
	//
	// The weights buffer works differently compared to the MLP:
	//   o its 4-dimension
	//   o the first 2 indices address the source and the destination
	//     layer matrix, respectively.
	//   o the last 2 indices address the connections in the usual manner
	// 
	// For instance, this.weights[0][1][4][6] addresses connection from
	// neuron 4 of layer 0 to neuron 6 of layer 1. 
	//
	// Note that in this implementation only the forward weight matrices [l][l+1]
	// and and recurrent weight matrices [l][l] (for non-input non-output layers)
	// are defined -> all others are null. (i.e. only simple recurrences withinn each layer)
	//
	private double[][][][] weights; // [fromLayer][toLayer][fromLIndex][toLIndex]
	private double[][][][] dweights; // [fromLayer][toLayer][fromLIndex][toLIndex]

	private int bufferlength    = 0;
	private int lastinputlength = 0;

	public boolean getBias(final int layer) {
		assert(layer > 0 && layer < this.layersnum);
		return this.usebias[layer];
	}

	/**
	 * Constructor of the RNN class. The function signature ensures that there is
	 * at least an input and an output layer (in this case layern.length would be 0). 
	 * Otherwise layer2 would be the first hidden layer and layern would contain
	 * at least the output layer and, optionally, further hidden layers. Effectively,
	 * the firstly given layer defines the input layer size, the lastly given number
	 * defines the output layer size, and the numbers in between define the hidden 
	 * layers sizes, accordingly.
	 * 
	 * @param input Number of input neurons.
	 * @param hiddenLayerSizes number of hidden layers and their sizes
	 * @param outputSize size of output layer
	 * @param useBiases specifies which layer uses biases - index 0 = input-to-first-hidden-layer.
	 */
	public ANNLayer_RNN(
			final int inputSize, 
			final int[] hiddenLayerSizes, 
			final int outputSize, 
			final boolean[] useBiases,
			final ACT_FUNCT[] actFunctions
			) {
		//
		this.inputSize = inputSize;
		this.layerSizes = ANNLayer_RNN.join(inputSize, hiddenLayerSizes, outputSize);
		this.layersnum = this.layerSizes.length; // number of layers including input and output layer
		this.outputlayer = this.layersnum - 1;
		this.outputSize = outputSize;
		//
		// set up buffers.
		//
		this.net      = new double[this.layersnum][][];
		this.act      = new double[this.layersnum][][];
		this.delta    = new double[this.layersnum][][];
		this.bwbuffer = new double[this.layersnum][][];
		//
		this.rebufferOnDemand(1);
		//
		this.usebias  = useBiases.clone();
		this.actFuncts = actFunctions.clone();
		//
		this.weights  = new double[this.layersnum][this.layersnum][][];
		this.dweights = new double[this.layersnum][this.layersnum][][];
		//
		int sumweights = 0;
		//
		for (int fromLayer = 0; fromLayer < this.layersnum-1; fromLayer++) {
			int toLayer = fromLayer+1;
			// forward weights... 
			if(this.usebias[fromLayer]) {
				this.weights[fromLayer][toLayer]  = new double[this.layerSizes[fromLayer] + 1][this.layerSizes[toLayer]];
				this.dweights[fromLayer][toLayer] = new double[this.layerSizes[fromLayer] + 1][this.layerSizes[toLayer]];
				sumweights += (this.layerSizes[fromLayer] + 1) * (this.layerSizes[toLayer]);
			}else{
				this.weights[fromLayer][toLayer]  = new double[this.layerSizes[fromLayer]][this.layerSizes[toLayer]];
				this.dweights[fromLayer][toLayer] = new double[this.layerSizes[fromLayer]][this.layerSizes[toLayer]];
				sumweights += (this.layerSizes[fromLayer]) * (this.layerSizes[toLayer]);				
			}
			//
			// if the current layer is a hidden layer, also add recurrent connections.
			//
			if (toLayer < this.layersnum-1) {
				// recurrent weights within a hidden layer - note: there are no bias weights.
				this.weights[toLayer][toLayer]  = new double[this.layerSizes[toLayer]][this.layerSizes[toLayer]];
				this.dweights[toLayer][toLayer] = new double[this.layerSizes[toLayer]][this.layerSizes[toLayer]];
				//
				sumweights += (this.layerSizes[toLayer] * this.layerSizes[toLayer]);
			}
			//
		}
		//
		this.numWeights = sumweights;
	}

	private static int[] join(final int inputSize, final int[] hiddenSizes, final int outputSize) {
		final int[] result = new int[2 + hiddenSizes.length];
		result[0] = inputSize;
		for (int i = 0; i < hiddenSizes.length; i++) {
			result[i + 1] = hiddenSizes[i];
		}
		result[hiddenSizes.length+1] = outputSize;
		return result;
	}

	public void resetAllActivitiesToZero() {
		for (int l = 0; l < this.layersnum; l++) {
			for (int i = 0; i < this.layerSizes[l]; i++) {
				for (int t = 0; t < this.bufferlength; t++) {
					this.act[l][i][t]      = 0.0;
					if (l > 0) {
						this.net[l][i][t]      = 0.0;
						this.delta[l][i][t]    = 0.0;
						this.bwbuffer[l][i][t] = 0.0;
					}
				}
			}
		}
		//
		this.lastinputlength = 0;
	}

	/**
	 * Resets the first net activities in the activity buffer, 
	 * in order to prevent recurrent connections from yielding an influence. 
	 */
	public void resetFirstTimeStepActivitiesToZero() {
		for (int l = 0; l < this.layersnum; l++) {
			for (int i = 0; i < this.layerSizes[l]; i++) {
				this.act[l][i][0] = 0.0;
			}
		}
	}

	/**
	 * Reallocating memory to store all the activities of a forward pass through a time series. 
	 * 
	 * @param sequencelength The length of the upcoming time series.
	 */
	public void rebufferOnDemand(final int sequencelength) {
		//
		if (this.bufferlength != sequencelength) {
			for (int l = 0; l < this.layersnum; l++) {
				this.net[l]      = new double[this.layerSizes[l]][sequencelength];
				this.delta[l]    = new double[this.layerSizes[l]][sequencelength];
				this.bwbuffer[l] = new double[this.layerSizes[l]][sequencelength];
				this.act[l] = new double[this.layerSizes[l]][sequencelength];
			}
		}
		//
		this.bufferlength    = sequencelength;
		this.lastinputlength = 0;
	}


	/**
	 * Computes the forward pass, i.e., propagates an input 
	 * vector through the network to the output layer. 
	 * This method does so for one time step online... 
	 * .. assuming the recurrence unfolds in the first time step
	 * 
	 * @param input Input vector.
	 * @return Output vector.
	 */
	public double[] forwardPass(final double[] input) {
		assert(input.length == this.inputSize);
		// execute on self-recurrent step.
		this.oneForwardPassAt(input, 0, 0);
		// store & return output.
		final double[] output  = new double[this.outputSize];
		final int outputlayer    = this.layersnum - 1;
		for (int k = 0; k < this.outputSize; k++) {
			output[k] = this.act[outputlayer][k][0];
		}
		return output;
	}


	/**
	 * Computes the forward pass, i.e., propagates a sequence
	 * of input vectors through the network to the output layer.
	 * Thereby, the activity at time 0 is assumed to correspond to the initial internal network activity, 
	 *  such that there are t+1 activity buffers, 
	 *  whereby the first output is generate by butter 1 and the last one by buffer t.
	 *  
	 * @param input Sequence of input vectors.
	 * @return Output Sequence of output vectors.
	 */
	public double[][] forwardPassZero(final double[][] input) {
		//
		assert(input.length == this.inputSize);
		//
		final int sequencelength = input.length;
		assert(bufferlength >= sequencelength+1); // disallow not processing the complete sequence!
		final int outputlayer    = this.layersnum - 1;
		// starting at t=1 since t=0 are the initialization values. 
		for (int t = 1; t < sequencelength+1; t++) {
			this.oneForwardPassAt(input[t-1], t, t-1);
		}
		// store output.
		final double[][] output  = new double[sequencelength][this.outputSize];
		for (int t = 0; t < sequencelength; t++) {
			for (int k = 0; k < this.outputSize; k++) {
				output[t][k] = this.act[outputlayer][k][t+1];
			}
		}
		//
		// Store input length of the current sequence.
		this.lastinputlength = sequencelength;
		//
		// return output layer activation as output.
		//
		return output;
	}

	/**
	 * Executes one forward pass time step at time t in the sequence
	 * assuming that the previous hidden neural state activities were stored at time step prevt.
	 * (note: prevt may be equal to t)
	 * 
	 * @param input the fed-in input.
	 * @param t the current time step in a time series
	 * @param prevt the previous time step where the initial hidden activities (of the previous time step) are stored.
	 */
	public void oneForwardPassAt(double[] input, int t, int prevt) {
		for (int i = 0; i < input.length; i++) {
			this.act[0][i][t] = input[i];
		}
		// compute output layer-wise. start with the first
		// hidden layer (or the outputlayer if there is no
		// hidden layer).
		final int outputlayer    = this.layersnum - 1;
		for (int fromLayer = 0; fromLayer < this.layersnum-1; fromLayer++) {
			// first compute all the net (integrate the inputs) values and activations.
			int toLayer = fromLayer+1;
			final int fromLayerSize = this.layerSizes[fromLayer];
			final int toLayerSize    = this.layerSizes[toLayer];
			final double[][] ff_weights = this.weights[fromLayer][toLayer];
			final double[][] fb_weights = this.weights[toLayer][toLayer];
			//
			for (int j = 0; j < toLayerSize; j++) {
				// initialize netjt with zero or the weighted bias.
				double netjt = 0;
				if (this.usebias[fromLayer]) {
					netjt = BIAS * ff_weights[fromLayerSize][j];
				}
				// integrate feed-forward input.
				for (int i = 0; i < fromLayerSize; i++) {
					netjt += this.act[fromLayer][i][t] * ff_weights[i][j];
				}
				//
				if (toLayer < outputlayer) {
					// integrate recurrent input.
					for (int i = 0; i < toLayerSize; i++) {
						netjt += this.act[toLayer][i][prevt] * fb_weights[i][j];
					}
				}
				//  
				this.net[toLayer][j][t] = netjt;
			}
			// now we compute the activations of the neurons in the current layer.
			// tanh hidden layer.
			for (int j = 0; j < toLayerSize; j++) {
				this.act[toLayer][j][t] = ActivationFunctions.getFofX(this.net[toLayer][j][t], this.actFuncts[fromLayer]);//ActivationFunctions.tanh(this.net[toLayer][j][t]);
			}
		}
	}

	/**
	 * Computes one backward pass step through the network at time index t. 
	 * Surrounded by the previous time index and next time index. 
	 * ASSUMES that the bwbuffer for this time point t has been set appropriately (from somewhere)
	 * 
	 * @param target the target values at this time index - might be null, indicating that no target values are set here
	 * @param t the time index
	 * @param prevt previous time index (usually t-1) - -1 indicates that t is the first time index in a forward pass series
	 * @param nextt next time index (usually t+1) - -1 indicates that t is the last time index in a forward pass series
	 */
	@Override
	public void oneBackwardPassAt(int t, int prevt, int nextt) {
		// inject the output/target discrepancy into this.delta. Note that
		// this.delta functions analogously to this.net but is used to
		// store into "back-flowing" inputs (deltas).
		for (int i = 0; i < this.delta[outputlayer].length; i++) {
			this.delta[outputlayer][i][t] = 
					ActivationFunctions.getDerivativeFofX(this.net[outputlayer][i][t], this.actFuncts[outputlayer-1]) *
					this.bwbuffer[outputlayer][i][t];
		}
		// back-propagate the error through the network -- we compute the deltas --
		// starting with the output layer.
		// ending with the toLayer two before the input layer, thus the fromLayer one before the input layer(no deltas for the input layer)
		for(int toLayer = outputlayer; toLayer > 1; toLayer--) {
			int fromLayer = toLayer-1;
			int fromLayerSize = this.layerSizes[fromLayer];
			int toLayerSize = this.layerSizes[toLayer];
			final double[][] ff_weights = this.weights[fromLayer][toLayer];
			final double[][] fb_weights = this.weights[fromLayer][fromLayer];
			for(int i=0; i<fromLayerSize; i++) {
				double dij = 0;
				for(int j=0; j<toLayerSize; j++) {
					dij += ff_weights[i][j]*this.delta[toLayer][j][t];
				}
				if(nextt>=0) { // recurrence within hidden layers
					for(int j=0; j<fromLayerSize; j++)
						dij += fb_weights[i][j] * this.delta[fromLayer][j][nextt];
				}
				this.bwbuffer[fromLayer][i][t] = dij;
				this.delta[fromLayer][i][t] = 
						ActivationFunctions.getDerivativeFofX(this.net[fromLayer][i][t], this.actFuncts[fromLayer-1]) 
						* dij;
			}
		}
		// finally project the bw also back to the input.
		int toLayerSize = this.layerSizes[1];
		final double[][] ff_weights = this.weights[0][1];
		for(int i=0; i<this.inputSize; i++) {
			double dij = 0;
			for(int j=0; j<toLayerSize; j++) {
				dij += ff_weights[i][j] * this.delta[1][j][t];
			}
			this.bwbuffer[0][i][t] = dij;
		}		
	}

	/**
	 * Executes a backwards error pass through the network... 
	 * 
	 * ASSUMES: that there is at least one target value.

	 * @param target
	 */
	public void backwardPassZero(final double[][] target) {
		//
		final int steps       = this.lastinputlength;
		//
		int t_target = target.length - 1; // can be used to separate target length from sequence length... (e.g. if only the last n targets are defined).
		assert(t_target > 0);
		// compute reversely in time.
		// starting at steps (NOT steps-1) due to the zero offset!
		// first case is "special" in that the next time step index must be set to -1
		//
		for (int t = steps; t >= 0; t--,t_target--) {
			if(t_target >= 0) {
				for (int j = 0; j < this.outputSize; j++) {
					this.bwbuffer[outputlayer][j][t] = (this.act[this.outputlayer][j][t] - target[t_target][j]);
				}
				if(t == steps) // first backprop. through time step
					this.oneBackwardPassAt(t, t-1, -1);
				else
					this.oneBackwardPassAt(t, t-1, t+1);
			}else{
				// If the target length is shorter than the input length, 
				// then the targets are the last outputs of the input sequence.
				for (int j = 0; j < this.outputSize; j++) {
					this.bwbuffer[outputlayer][j][t] = 0;
				}
				this.oneBackwardPassAt(t, t-1, t+1);				
			}
		}
		// 
		// Compute the weights derivatives.
		//
		setWeightsDerivativesGeneral(0, steps);
	}

	/**
	 * This is the general method that can process also weight derivatives wheren tInit>tLast,
	 * that is, when the indices of the last time sequence online went over the buffer (buffer is circular)
	 * @param tInit
	 * @param tLast
	 */
	public void setWeightsDerivativesGeneral(int tInit, int tLast) {
		if(tInit <= tLast) {
			setWeightsDerivatives(tInit, tLast);
			return;
		}
		final int outputlayer = this.act.length - 1;
		for (int toLayer = 1; toLayer <= outputlayer; toLayer++) {       
			// compute weights derivatives between previous layer and current layer.
			final int fromLayer = toLayer-1;
			int fromLayerSize = this.layerSizes[fromLayer];
			int toLayerSize = this.layerSizes[toLayer];
			final double[][] ff_dweights = this.dweights[fromLayer][toLayer];
			for(int i=0; i<fromLayerSize; i++) {
				for(int j=0; j<toLayerSize; j++) {
					double dw = 0;
					for(int t=tInit; t<this.bufferlength; t++) {
						dw += this.act[fromLayer][i][t] * this.delta[toLayer][j][t];
					}
					for(int t=0; t<tLast+1; t++) {
						dw += this.act[fromLayer][i][t] * this.delta[toLayer][j][t];
					}
					ff_dweights[i][j] = dw;
				}
			}
			// compute weights derivatives between bias and current layer.
			if(this.usebias[fromLayer]) {
				for(int j=0; j<act[toLayer].length; j++) {
					double dw = 0;
					for(int t=tInit; t<this.bufferlength; t++) {
						dw += this.delta[toLayer][j][t];
					}
					for(int t=0; t<tLast+1; t++) {
						dw += this.delta[toLayer][j][t];
					}
					ff_dweights[fromLayerSize][j] = dw;
				}
			}
			// compute recurrent weight derivatives between current layer and current layer.
			if(toLayer < outputlayer) {// recurrent connections within hidden layers only
				final double[][] fb_dweights = this.dweights[toLayer][toLayer];
				for(int i=0; i<toLayerSize; i++) {
					for(int j=0; j<toLayerSize; j++) {
						double dw = 0;
						for(int t=tInit; t<this.bufferlength-1; t++) {
							dw += this.act[toLayer][i][t] * this.delta[toLayer][j][t+1];
						}
						dw += this.act[toLayer][i][this.bufferlength-1] * this.delta[toLayer][j][0];
						for(int t=0; t<tLast; t++) {
							dw += this.act[toLayer][i][t] * this.delta[toLayer][j][t+1];
						}
						fb_dweights[i][j] = dw;
					}
				}
			}
		}

	}

	/** set the weights derivatives for a certain range of time stamps. 
	 * 
	 * @param tInit the time stamp where the error processing starts.
	 * @param tLast the time stamp up to which the forward pass processed the information and 
	 * 			according backwards pass error was processed.
	 */
	private void setWeightsDerivatives(int tInit, int tLast) {
		assert(tInit<tLast);
		//
		final int outputlayer = this.act.length - 1;
		for (int toLayer = 1; toLayer <= outputlayer; toLayer++) {       
			// compute weights derivatives between previous layer and current layer.
			final int fromLayer = toLayer-1;
			int fromLayerSize = this.layerSizes[fromLayer];
			int toLayerSize = this.layerSizes[toLayer];
			final double[][] ff_dweights = this.dweights[fromLayer][toLayer];
			for(int i=0; i<fromLayerSize; i++) {
				for(int j=0; j<toLayerSize; j++) {
					double dw = 0;
					for(int t=tInit; t<tLast+1; t++) {
						dw += this.act[fromLayer][i][t] * this.delta[toLayer][j][t];
					}
					ff_dweights[i][j] = dw;
				}
			}
			// compute weights derivatives between bias and current layer.
			if(this.usebias[fromLayer]) {
				for(int j=0; j<act[toLayer].length; j++) {
					double dw = 0;
					for(int t=tInit; t<tLast+1; t++) {
						dw += this.delta[toLayer][j][t];
					}
					ff_dweights[fromLayerSize][j] = dw;
				}
			}
			// compute recurrent weight derivatives between current layer and current layer.
			if(toLayer < outputlayer) {// recurrent connections within hidden layers only
				final double[][] fb_dweights = this.dweights[toLayer][toLayer];
				for(int i=0; i<toLayerSize; i++) {
					for(int j=0; j<toLayerSize; j++) {
						double dw = 0;
						for(int t=tInit; t<tLast; t++) {
							dw += this.act[toLayer][i][t] * this.delta[toLayer][j][t+1];
						}
						fb_dweights[i][j] = dw;
					}
				}
			}
		}
	}

	/**
	 * This method adapts the internal hidden state activity to tune the sequence
	 * towards generating the desired target ouput after depth steps. 
	 * 
	 * @param target desired target output
	 * @param depth steps after which the output is intended to be produced.
	 * @param numEpochs number of "epochs" the internal state is adapted
	 * @param learningRate of internal state adaptation
	 * @param momentum adaptation-rate momentum.
	 */
	public void forwardBackwardProject(
			final double[] target, 
			int depth, 
			int numEpochs, 
			double learningRate, 
			double momentum,
			ErrorBasedLearner.WEIGHT_UPDATE activeInferenceUpdateType
			) {
		//
		this.rebufferOnDemand(depth+1);
		double[][] zeroInput = new double[depth][this.inputSize];
		// initialize the adaptive activity adaptation mechanism
		ErrorBasedLearner ebl = null;
		final int numHiddenActivities = this.getHiddenActivitiesNum();

		switch(activeInferenceUpdateType) {
		case GradientDescent:
			ebl = new GradientDescentLearner(numHiddenActivities, learningRate, momentum);
			break;
		case Adam:
			ebl = new AdamLearner(numHiddenActivities, learningRate);
			break;
		}
		double initialActivities[] = new double[numHiddenActivities];
		ebl.resetValues(initialActivities);		
		double bwErrors[] = new double[numHiddenActivities];

		// begin of adaptation process
		for(int e=0; e<numEpochs; e++) {
			// note: internal activity of last epoch is intentionally used here as the internal start state
			// However, might want to adjust this to enforce an exact initial internal state. 
			this.forwardPassZero(zeroInput); 
			this.backwardPassZero(new double[][]{target});
			// now adapt the initial internal, hidden activity of the network!
			this.readBWErrors(bwErrors, 0);
			double newActivities[] = ebl.learningIteration(bwErrors);
			this.writeInternalActivities(newActivities, 0);
		}
		double[] output = getOutputActivities(lastinputlength);
		System.out.println("ERR: "+Tools.RMSE(output, target));
		// done... thus moving the hidden state of activation towards the aimed-at state.
		for(int t=0; t<depth; t++)
			this.forwardPass(zeroInput[0]);	// moving the hidden activations towards the aimed-at start activations.	
	}

	
	@Override
	public void getInternalActivityBounds(double[] lowerBounds,  double[] upperBounds) {
		int idx=0;
		for(int l=1; l<this.layersnum-1; l++) {
			for(int i=0; i<this.layerSizes[l]; i++) {
				lowerBounds[idx] = ActivationFunctions.getLowerBound(this.actFuncts[l-1]);
				upperBounds[idx] = ActivationFunctions.getUpperBound(this.actFuncts[l-1]);
				idx++;
			}
		}
	}

	/**
	 * Reads the internal activities that are relevant for inference of an 
	 * optimal hidden state initialization.
	 * 
	 * @param activities
	 */
	@Override
	public void readInternalActivities(double[] activities, int timePoint) {
		int idx=0;
		for(int l=1; l<this.layersnum-1; l++) {
			for(int i=0; i<this.layerSizes[l]; i++) {
				activities[idx++] = this.act[l][i][timePoint];
			}
		}
	}
	
	@Override
	public void readBWErrors(double[] bwErrors, int timePoint) {
		int idx=0;
		for(int l=1; l<this.layersnum-1; l++) {
			for(int i=0; i<this.layerSizes[l]; i++) {
				bwErrors[idx] = this.bwbuffer[l][i][timePoint];
				idx++;
			}
		}
	}

	@Override
	public void writeInternalActivities(double[] activities, int timePoint) {
		int idx=0;
		for(int l=1; l<this.layersnum-1; l++) {
			for(int i=0; i<this.layerSizes[l]; i++) {
				this.act[l][i][timePoint] = activities[idx];
				idx++;
			}
		}
	}
	
	/**
	 * Initializes the weights randomly and normal distributed with
	 * std. dev. 0.1.
	 * @param rnd Instance of Random.
	 */
	public void initializeWeights(final Random rnd, final double stddev) {
		for (int l1 = 0; l1 < this.weights.length; l1++) {
			for (int l2 = 0; l2 < this.weights[l1].length; l2++) {
				double[][] wll = this.weights[l1][l2];
				if (wll != null) {
					for (int i = 0; i < wll.length; i++) {
						for (int j = 0; j < wll[i].length; j++) {
							wll[i][j] = rnd.nextGaussian() * stddev;
						}
					}
				}
			}
		}
	}

	@Override
	public int getNumWeights() {
		return this.numWeights;
	}

	@Override
	public int getHiddenActivitiesNum() {
		int ret = 0;
		for(int l=1; l<this.layersnum-1; l++) {
			ret += this.layerSizes[l];
		}
		return ret;
	}

	@Override
	public int getOutputSize() {
		return this.outputSize;
	}
	
	@Override
	public int getInputSize() {
		return this.inputSize;
	}

	public void writeWeights(final double[] weights) {
		Tools.map(weights, this.weights);
	}

	public void readWeights(final double[] weights) {
		Tools.map(this.weights, weights);
	}

	public void readDiffWeights(final double[] myDiffWeights) {
		Tools.map(this.dweights, myDiffWeights);
	}

	/**
	 * Stochastic gradient descent.
	 * @param rnd Instance of Random.
	 * @param input Input vectors.
	 * @param target Target vectors.
	 * @param epochs Number of epochs.
	 * @param learningrate Value for the learning rate.
	 * @param momentumrate Value for the momentum rate.
	 * @param listener Listener to observe the training progress.
	 * @return The final epoch error.
	 */    
	public double trainStochasticIterative(
			final Random rnd, 
			final double[][][] input, 
			final double target[][][],
			final double epochs,
			final double learningRate,
			final double momentumRate,
			final LearningListener listener,
			final WEIGHT_UPDATE weightUpdateType
			) {
		//
		assert(input.length == target.length);
		//
		final double[] initialWeights       = new double[this.numWeights];
		final double[] weights       = new double[this.numWeights];
		final double[] dweights      = new double[this.numWeights];
		//
		this.readWeights(weights);
		//
		ErrorBasedLearner ebl = null;
		switch(weightUpdateType) {
		case GradientDescent:
			ebl = new GradientDescentLearner(this.numWeights, learningRate, momentumRate);
			break;
		case Adam:
			ebl = new AdamLearner(this.numWeights, learningRate, 0.9, 0.999);
			break;
		}
		this.readWeights(initialWeights);
		ebl.resetValues(initialWeights);
		//
		// create initial index permutation.
		//
		final int[] indices = new int[input.length];
		for (int i = 0; i < indices.length; i++) {
			indices[i] = i;
		}
		//
		double error = Double.POSITIVE_INFINITY;
		//
		// epoch loop.
		//
		for (int i = 0; i < epochs; i++) {
			//
			// shuffle indices.
			//
			Tools.shuffle(indices, rnd);
			//
			double errorsum = 0.0;
			//
			// train all samples in online manner, i.e. iterate over all samples
			// while considering the shuffled order and update the weights 
			// immediately after each sample
			//
			for(int j=0; j<indices.length; j++) {
				if(i+1 == epochs && j+1 == indices.length) {
					System.out.println("Very last training instance.");
				}
				//                this.resetFirstActivities(); // when working with forwardPassZero, this is not necessary!
				double[][] output = this.forwardPassZero(input[indices[j]]);
				errorsum += Tools.MSE(output,  target[indices[j]]);
				this.backwardPassZero(target[indices[j]]);

				this.readDiffWeights(dweights);

				final double[] newWeights = ebl.learningIteration(dweights);

				this.writeWeights(newWeights);
			}           
			//
			error = errorsum / (double)(input.length);
			if (listener != null) listener.afterEpoch(i + 1, error);
			if(i+1 == epochs) {
				System.out.println("Done with training");
			}
		}
		//
		return error;
	}

	@Override
	public double[] getOutputActivities(int t) {
		final double[] output = new double[this.outputSize];
		for (int k = 0; k < this.outputSize; k++) {
			output[k] = this.act[this.outputlayer][k][t];
		}
		return output;
	}

	@Override
	public double[] getInputBW(int t) {
		final double[] inputBW = new double[this.inputSize];
		for (int k = 0; k < this.inputSize; k++) {
			inputBW[k] = this.bwbuffer[0][k][t];
		}				
		return inputBW;
	}

	@Override
	public void writeNetwork(String fileName) {
		try{
			FileOutputStream f = new FileOutputStream(fileName);
			ObjectOutput s = new ObjectOutputStream(f);
			s.writeObject(this);
			s.close();
			f.close();
		}catch (IOException e) {
			e.printStackTrace();
		}	
	}

	public static ANNLayer_RNN readNetwork(String fileName) throws IOException, ClassNotFoundException, ClassCastException {
		FileInputStream f = new FileInputStream(fileName);
		ObjectInputStream s = new ObjectInputStream(f);
		ANNLayer_RNN rnn = (ANNLayer_RNN)s.readObject();
		s.close();
		f.close();
		return rnn;
	}

	@Override
	public void setOutputBW(double[] bwSignals, int timePoint) {
		for(int o=0; o<this.outputSize; o++) {
			this.bwbuffer[this.outputlayer][o][timePoint] = bwSignals[o];
		}
	}
	
	public double[][] getOutputActivityTS() {
		final double[][] output  = new double[lastinputlength][this.outputSize];
		int outputlayer = this.layersnum-1;
		for (int t = 0; t < lastinputlength; t++) {
			for (int k = 0; k < this.outputSize; k++) {
				output[t][k] = this.act[outputlayer][k][t+1];
			}
		}
		return output;
	}

	public double[][] getInputBWTS() {
		final double[][] inputBWs = new double[lastinputlength][this.inputSize];
		for (int t = 0; t < lastinputlength; t++) {
			for (int k = 0; k < this.inputSize; k++) {
				inputBWs[t][k] = this.bwbuffer[0][k][t];
			}				
		}
		return inputBWs;
	}


}