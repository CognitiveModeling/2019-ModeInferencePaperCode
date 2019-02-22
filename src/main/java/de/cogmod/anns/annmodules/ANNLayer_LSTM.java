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
import de.cogmod.anns.errorlearners.GradientDescentLearner;
import de.cogmod.anns.errorlearners.ErrorBasedLearner.WEIGHT_UPDATE;
import de.cogmod.utilities.ActivationFunctions;
import de.cogmod.utilities.LearningListener;
import de.cogmod.utilities.Tools;
import de.cogmod.utilities.ActivationFunctions.ACT_FUNCT;

public class ANNLayer_LSTM implements ANNLayer, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public final static double BIAS = 1.0;

	private final int inputSize;
	private final int hiddenSize;
	private final int cellSize;
	private final int outputSize;

	// input array buffers are in [inputSize][time]
	private double[][] actinput;
	private double[][] bwinput;

	// neural gate arrays are [0=input gate; 1=forget gate; 2=output gate][LSTM-cell-index][time]
	private double[][][] netgates; // input gate net input
	private double[][][] actgates; // input gate activity 
	// net input is in [LSTM-cell index][CEC cell index][time]
	private double[][][] netinp;
	private double[][][] actinpng; // not yet acted cell activation input
	private double[][][] actinpgated;
	// net input and activity for each with each cell in each CEC cell	
	//	[LSTM-cell index][CEC cell index][time]
	private double[][][] netcec;
	private double[][][] actcec;
	// net ouptu and activity for each with each cell in each CEC cell	
	//	[LSTM-cell index][CEC cell index][time]
	private double[][][] netout;
	private double[][][] actoutng;
	private double[][][] actoutgated;

	// ouptut array buffers are in [outputSize][time]
	private double[][] netoutput;
	private double[][] actoutput;
	private double[][] bwoutput;
	private double[][] deltaoutput;

	//	ACCORDING ERROR BUFFERS
	private double[][][] bwgates; 
	private double[][][] deltagates; 

	private double[][][] bwinpng;  // ng=not gated, i.e. bw value before multiplying the gate factor to it
	private double[][][] bwinpgated;  // bw value after multiplying the gate factor to the backwards error signal
	private double[][][] deltainp; 
	private double[][][] bwcec; 
	private double[][][] deltacec; 
	private double[][][] bwoutng; // ng=not gated, i.e. bw value before multiplying the gate factor to it
	private double[][][] bwoutgated;  // bw value after multiplying the gate factor to the backwards error signal
	private double[][][] deltaout; 

	private boolean[] usebias; // use bias in input-to-gates, input-to-CEC, and/or output? [0=input-to-gates; 1=input-to-cell; 2=output]

	private ACT_FUNCT outputActFunct;

	// The weight buffers 
	private double[][][] weightsGatesInput; // [to gate ig/hg/og][from-index(inputSize)][to-index(hiddenSize)]
	private double[][][][] weightsGatesHidden; // [to gate ig/hg/og][from-cell-index(hiddenSize)][from-CEC-cell-index][to-index(hiddenSize)]
	private double[][][] weightsCellInput; // [from-index(inputSize)][to-cell-index(hiddenSize)][to-CEC-cell-index(cellSize)]
	private double[][][][] weightsCellHidden; // [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][to-cell-index(hiddenSize)][to-CEC-cell-index(cellSize)]
	private double[][][] weightsPeepholes; // [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][ig/hg/og] 
	private double[][][] weightsOutput; //  [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][to-index(outputSize)]
	// The weight derivative buffers
	private double[][][] dweightsGatesInput;
	private double[][][][] dweightsGatesHidden;
	private double[][][] dweightsCellInput;
	private double[][][][] dweightsCellHidden;
	private double[][][] dweightsPeepholes;
	private double[][][] dweightsOutput; 

	private final int inputGateIndex = 0;
	private final int forgetGateIndex = 1;
	private final int outputGateIndex = 2;

	//	total number of weights in this layer
	private int numWeights;

	//current time buffer length.
	private int bufferlength    = 0;
	// length of last forward pass input time series.
	private int lastinputlength = 0;

	public ANNLayer_LSTM(final int inputSize, 
			final int hiddenSize, 
			final int numCECCells, 
			final int outputSize, 
			final boolean[] useInputBias,
			final ActivationFunctions.ACT_FUNCT outputActivationFunction
			) {
		//
		this.usebias = useInputBias.clone();

		this.outputActFunct = outputActivationFunction;

		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.cellSize = numCECCells;
		this.outputSize = outputSize;

		this.actinput = new double[inputSize][];
		this.bwinput = new double[inputSize][];

		this.netgates = new double[3][hiddenSize][];
		this.actgates = new double[3][hiddenSize][];

		this.netinp = new double[hiddenSize][cellSize][];
		this.actinpng = new double[hiddenSize][cellSize][];
		this.actinpgated = new double[hiddenSize][cellSize][];
		this.netcec = new double[hiddenSize][cellSize][];
		this.actcec = new double[hiddenSize][cellSize][];
		this.netout = new double[hiddenSize][cellSize][];
		this.actoutng = new double[hiddenSize][cellSize][];
		this.actoutgated = new double[hiddenSize][cellSize][];

		this.bwgates = new double[3][hiddenSize][];
		this.deltagates = new double[3][hiddenSize][];

		this.bwinpng = new double[hiddenSize][cellSize][];
		this.bwinpgated = new double[hiddenSize][cellSize][];
		this.deltainp = new double[hiddenSize][cellSize][];
		this.bwcec = new double[hiddenSize][cellSize][];
		this.deltacec = new double[hiddenSize][cellSize][];
		this.bwoutng = new double[hiddenSize][cellSize][];
		this.bwoutgated = new double[hiddenSize][cellSize][];
		this.deltaout = new double[hiddenSize][cellSize][];

		this.netoutput = new double[outputSize][];
		this.actoutput = new double[outputSize][];
		this.bwoutput = new double[outputSize][];
		this.deltaoutput = new double[outputSize][];
		//
		this.rebufferOnDemand(1);
		//		
		this.weightsGatesInput = new double[3][][];
		this.dweightsGatesInput = new double[3][][];
		this.weightsGatesHidden = new double[3][][][];
		this.dweightsGatesHidden = new double[3][][][];
		int sumweights = 0;
		if(this.usebias[0]) {
			for(int i=0; i<3; i++) {
				this.weightsGatesInput[i] = new double[inputSize+1][hiddenSize];
				this.dweightsGatesInput[i] = new double[inputSize+1][hiddenSize];
				sumweights += (inputSize+1) * hiddenSize;
			}
		}else{
			for(int i=0; i<3; i++) {
				this.weightsGatesInput[i] = new double[inputSize][hiddenSize];
				this.dweightsGatesInput[i] = new double[inputSize][hiddenSize];
				sumweights += inputSize * hiddenSize;
			}
		}
		for(int i=0; i<3; i++) {
			this.weightsGatesHidden[i] = new double[hiddenSize][cellSize][hiddenSize];
			this.dweightsGatesHidden[i] = new double[hiddenSize][cellSize][hiddenSize];
			sumweights += hiddenSize * cellSize * hiddenSize;
		}

		if(this.usebias[1]) {
			this.weightsCellInput = new double[inputSize+1][hiddenSize][cellSize];
			this.dweightsCellInput = new double[inputSize+1][hiddenSize][cellSize];
			sumweights += (inputSize+1) * hiddenSize * cellSize;
		}else{
			this.weightsCellInput = new double[inputSize][hiddenSize][cellSize];
			this.dweightsCellInput = new double[inputSize][hiddenSize][cellSize];
			sumweights += inputSize * hiddenSize * cellSize;
		}
		this.weightsCellHidden = new double[hiddenSize][cellSize][hiddenSize][cellSize];
		this.dweightsCellHidden = new double[hiddenSize][cellSize][hiddenSize][cellSize];
		sumweights += hiddenSize * cellSize * hiddenSize * cellSize;

		this.weightsPeepholes = new double[hiddenSize][cellSize][3];
		this.dweightsPeepholes = new double[hiddenSize][cellSize][3];
		sumweights += hiddenSize * cellSize * 3; 	

		if(this.usebias[2]) {
			this.weightsOutput = new double[hiddenSize+1][cellSize][outputSize];
			this.dweightsOutput = new double[hiddenSize+1][cellSize][outputSize];
			sumweights += hiddenSize * cellSize * outputSize + outputSize; // note: bias weights are in [hiddenSize][0][index-output]			
		}else{
			this.weightsOutput = new double[hiddenSize][cellSize][outputSize];
			this.dweightsOutput = new double[hiddenSize][cellSize][outputSize];
			sumweights += hiddenSize * cellSize * outputSize; 			
		}
		this.numWeights = sumweights;
	}

	/**
	 * Reallocating memory to store all the activities of a forward pass through a time series. 
	 * 
	 * @param sequencelength The length of the upcoming time series.
	 */
	@Override
	public void rebufferOnDemand(final int sequencelength) {
		assert(sequencelength>0);
		//
		if (this.bufferlength != sequencelength) {
			for(int i=0; i<inputSize; i++) {
				this.actinput[i] = new double[sequencelength];
				this.bwinput[i] = new double[sequencelength];
			}
			for(int i=0; i<hiddenSize; i++) {
				for(int g=0; g<3; g++) {
					this.netgates[g][i] = new double[sequencelength];
					this.actgates[g][i] = new double[sequencelength];
					this.bwgates[g][i] = new double[sequencelength];
					this.deltagates[g][i] = new double[sequencelength];
				}
				for(int c=0; c<this.cellSize; c++) {
					this.netinp[i][c] = new double[sequencelength];
					this.actinpng[i][c] = new double[sequencelength];
					this.actinpgated[i][c] = new double[sequencelength];
					this.netcec[i][c] = new double[sequencelength];
					this.actcec[i][c] = new double[sequencelength];
					this.netout[i][c] = new double[sequencelength];
					this.actoutng[i][c] = new double[sequencelength];
					this.actoutgated[i][c] = new double[sequencelength];

					this.bwinpng[i][c] = new double[sequencelength];
					this.bwinpgated[i][c] = new double[sequencelength];
					this.deltainp[i][c] = new double[sequencelength];
					this.bwcec[i][c] = new double[sequencelength];
					this.deltacec[i][c] = new double[sequencelength];
					this.bwoutng[i][c] = new double[sequencelength];
					this.bwoutgated[i][c] = new double[sequencelength];
					this.deltaout[i][c] = new double[sequencelength];
				}
			}
			for(int i=0; i<outputSize; i++) {
				this.netoutput[i] = new double[sequencelength];
				this.actoutput[i] = new double[sequencelength];
				this.bwoutput[i] = new double[sequencelength];
				this.deltaoutput[i] = new double[sequencelength];				
			}
		}
		//		
		this.bufferlength    = sequencelength;
		this.lastinputlength = 0;
	}

	/**
	 * Resets all neural network activities to zero.  
	 * i.e. net activities, actual activities, bw errors, and delta errors.
	 */
	public void resetAllActivitiesToZero() {
		for(int i=0; i<inputSize; i++) {
			for(int t=0; t<this.bufferlength; t++) {
				this.actinput[i][t]=0;
				this.bwinput[i][t]=0;
			}
		}
		for(int i=0; i<hiddenSize; i++) {
			for(int t=0; t<this.bufferlength; t++) {
				for(int g=0; g<3; g++) {
					this.netgates[g][i][t] = 0;
					this.actgates[g][i][t] = 0;
					this.bwgates[g][i][t] = 0;
					this.deltagates[g][i][t] = 0;
				}

				for(int c=0; c<this.cellSize; c++) {
					this.netinp[i][c][t] = 0;
					this.actinpng[i][c][t] = 0;
					this.actinpgated[i][c][t] = 0;
					this.netcec[i][c][t] = 0;
					this.actcec[i][c][t] = 0;
					this.netout[i][c][t] = 0;
					this.actoutng[i][c][t] = 0;
					this.actoutgated[i][c][t] = 0;

					this.bwinpng[i][c][t] = 0;
					this.bwinpgated[i][c][t] = 0;
					this.deltainp[i][c][t] = 0;
					this.bwcec[i][c][t] = 0;
					this.deltacec[i][c][t] = 0;
					this.bwoutng[i][c][t] = 0;
					this.bwoutgated[i][c][t] = 0;
					this.deltaout[i][c][t] = 0;
				}
			}
		}
		for(int i=0; i<outputSize; i++) {
			for(int t=0; t<this.bufferlength; t++) {
				this.netoutput[i][t] = 0;
				this.actoutput[i][t] = 0;
				this.bwoutput[i][t] = 0;
				this.deltaoutput[i][t] = 0;
			}
		}
		this.lastinputlength = 0;
	}

	/**
	 * Resets all neural network activities to zero.  
	 * i.e. net activities, actual activities, bw errors, and delta errors.
	 */
	public void resetFirstTimeStepActivitiesToZero() {
		for(int i=0; i<inputSize; i++) {
			this.actinput[i][0]=0;
			this.bwinput[i][0]=0;
		}
		for(int i=0; i<hiddenSize; i++) {
			for(int g=0; g<3; g++) {
				this.netgates[g][i][0] = 0;
				this.actgates[g][i][0] = 0;
				this.bwgates[g][i][0] = 0;
				this.deltagates[g][i][0] = 0;
			}

			for(int c=0; c<this.cellSize; c++) {
				this.netinp[i][c][0] = 0;
				this.actinpng[i][c][0] = 0;
				this.actinpgated[i][c][0] = 0;
				this.netcec[i][c][0] = 0;
				this.actcec[i][c][0] = 0;
				this.netout[i][c][0] = 0;
				this.actoutng[i][c][0] = 0;
				this.actoutgated[i][c][0] = 0;

				this.bwinpng[i][c][0] = 0;
				this.bwinpgated[i][c][0] = 0;
				this.deltainp[i][c][0] = 0;
				this.bwcec[i][c][0] = 0;
				this.deltacec[i][c][0] = 0;
				this.bwoutng[i][c][0] = 0;
				this.bwoutgated[i][c][0] = 0;
				this.deltaout[i][c][0] = 0;
			}
		}
		for(int i=0; i<outputSize; i++) {
			this.netoutput[i][0] = 0;
			this.actoutput[i][0] = 0;
			this.bwoutput[i][0] = 0;
			this.deltaoutput[i][0] = 0;
		}		
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
		this.lastinputlength=1;
		return getOutputActivities(0);
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
	private double[][] forwardPassZero(final double[][] input) {
		assert(input[0].length == this.inputSize);

		final int sequencelength = input.length;
		assert(bufferlength >= sequencelength+1); // disallow not processing the complete sequence!

		// starting at t=1 since in t=0 are the initialization values. 
		for (int t = 1; t <= sequencelength; t++) {
			this.oneForwardPassAt(input[t-1], t, t-1);
		}
		// Store input length of the current sequence.
		this.lastinputlength = sequencelength;
		// store & return full output.
		final double[][] output  = new double[sequencelength][this.outputSize];
		for (int t = 0; t < sequencelength; t++) {
			for (int k = 0; k < this.outputSize; k++) {
				output[t][k] = this.actoutput[k][t+1];
			}
		}
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
	@Override
	public void oneForwardPassAt(double[] input, int t, int prevt) {
		for (int i = 0; i < input.length; i++) {
			this.actinput[i][t] = input[i];
		}
		for(int g = 0; g < 2; g++) { // input and forget gate processing
			for(int j = 0; j < this.hiddenSize; j++ ) {
				double netgjt = 0;
				if(this.usebias[0]) {
					netgjt = ANNLayer_LSTM.BIAS * this.weightsGatesInput[g][inputSize][j];
				}
				// input activations.
				for(int i=0; i < this.inputSize; i++) {
					netgjt += this.actinput[i][t] * this.weightsGatesInput[g][i][j];
				}
				// hidden recurrent activations
				for(int h=0; h < this.hiddenSize; h++) {
					for(int cfrom=0; cfrom < this.cellSize; cfrom++) {
						netgjt += this.actoutgated[h][cfrom][prevt] * this.weightsGatesHidden[g][h][cfrom][j];
					}
				}
				// peephole activations
				for(int c=0; c < this.cellSize; c++) {
					netgjt += this.actcec[j][c][prevt] * this.weightsPeepholes[j][c][g];
				}
				this.netgates[g][j][t] = netgjt;
				this.actgates[g][j][t] = ActivationFunctions.sigmoid(netgjt);
			}
		}
		// cell input processing
		for(int j=0; j < this.hiddenSize; j++) {
			for(int c=0; c < this.cellSize; c++) {
				double netjct = 0;
				if(this.usebias[1]) {
					netjct = ANNLayer_LSTM.BIAS * this.weightsCellInput[inputSize][j][c];
				}
				for(int i=0; i<this.inputSize; i++) {
					netjct += this.actinput[i][t] * this.weightsCellInput[i][j][c];
				}
				for(int h=0; h<this.hiddenSize; h++) {
					for(int cfrom=0; cfrom<this.cellSize; cfrom++)
						netjct += this.actoutgated[h][cfrom][prevt] * this.weightsCellHidden[h][cfrom][j][c];
				}
				this.netinp[j][c][t] = netjct;
				// act input into one cell in a CEC - -- taking the hyperbolic tangent from the net input
				this.actinpng[j][c][t] = ActivationFunctions.tanh(netjct);
				// then gating the activation yielding the input
				this.actinpgated[j][c][t] = this.actgates[this.inputGateIndex][j][t] * this.actinpng[j][c][t];
				// cell state combined from last cell output via forget gate and external input
				this.netcec[j][c][t] = this.actinpgated[j][c][t] + 
						this.actgates[this.forgetGateIndex][j][t] * this.actcec[j][c][prevt];
				this.actcec[j][c][t] = this.netcec[j][c][t]; // cec activation function is the identity
			}
		}
		// output gate (with respect to current cell states!)
		for(int j = 0; j < this.hiddenSize; j++ ) {
			double netjt = 0;
			if(this.usebias[0]) {
				netjt = ANNLayer_LSTM.BIAS * this.weightsGatesInput[this.outputGateIndex][inputSize][j];
			}
			// input activations.
			for(int i=0; i < this.inputSize; i++) {
				netjt += this.actinput[i][t] * this.weightsGatesInput[this.outputGateIndex][i][j];
			}
			// hidden recurrent activations
			for(int h=0; h < this.hiddenSize; h++) {
				for(int cfrom=0; cfrom < this.cellSize; cfrom++) {
					netjt += this.actoutgated[h][cfrom][prevt] * this.weightsGatesHidden[this.outputGateIndex][h][cfrom][j];
				}
			}
			// peephole activations
			for(int c=0; c < this.cellSize; c++) {
				netjt += this.actcec[j][c][t] * this.weightsPeepholes[j][c][this.outputGateIndex];
			}
			this.netgates[this.outputGateIndex][j][t] = netjt;
			this.actgates[this.outputGateIndex][j][t] = ActivationFunctions.sigmoid(netjt);
		}
		// cell output
		for(int j=0; j < this.hiddenSize; j++) {
			for(int c=0; c < this.cellSize; c++) {
				this.netout[j][c][t] = actcec[j][c][t];
				this.actoutng[j][c][t] = ActivationFunctions.tanh(netout[j][c][t]);
				this.actoutgated[j][c][t] = this.actoutng[j][c][t] * this.actgates[this.outputGateIndex][j][t];
			}
		}
		// to output layer
		for(int j=0; j < this.outputSize; j++) {
			double netj = 0;
			if(this.usebias[2]) {
				netj = ANNLayer_LSTM.BIAS * this.weightsOutput[hiddenSize][0][j];
			}
			for(int h=0; h < this.hiddenSize; h++) {
				for(int cfrom=0; cfrom < this.cellSize; cfrom++) {
					netj += this.actoutgated[h][cfrom][t] * this.weightsOutput[h][cfrom][j];
				}
			}
			this.netoutput[j][t] = netj;
			this.actoutput[j][t] = ActivationFunctions.getFofX(netj, this.outputActFunct);
		}
	}

	//	public static void main(String[] args) {
	//		ANNLayer_LSTM netTemp = new ANNLayer_LSTM(3, 
	//				5, 2, 4, 
	//				new boolean[]{false, false, false}, ACT_FUNCT.Tanh);
	//		double[] arr1 = new double[netTemp.getWeightsNum()];
	//		double[] arr2 = new double[netTemp.getWeightsNum()];
	//		Random rnd = new Random(42);
	//		netTemp.initializeWeights(rnd, 0.1);
	//		netTemp.readWeights(arr1);
	//		for(int i=0; i<arr1.length; i++)
	//			arr1[i] += rnd.nextGaussian();
	//		netTemp.writeWeights(arr1);
	//		netTemp.readWeights(arr2);
	//		boolean eq = Tools.equalsArray(arr1, arr2);
	//		if(eq) {
	//			System.out.println("They are equal! :-)");
	//		}else{
	//			System.err.println("NOT EQUAL :-(");
	//		}
	//	}



	/**
	 * Computes one backward pass step through the network at time index t. 
	 * Surrounded by the previous time index and next time index.
	 * Assumes that the bw signals in the output layer are set (starts with them).
	 * Assumes that all neural activities are set accordingly.
	 * 
	 * @param target the target values at this time index - might be null, indicating that no target values are set here
	 * @param t the time index
	 * @param prevt previous time index (usually t-1) - -1 indicates that t is the first time index in a forward pass series
	 * @param nextt next time index (usually t+1) - -1 indicates that t is the last time index in a forward pass series
	 */
	@Override
	public void oneBackwardPassAt(int t, int prevt, int nextt) {
		//
		for (int j = 0; j < this.outputSize; j++) {
			this.deltaoutput[j][t] = ActivationFunctions.getDerivativeFofX(this.netoutput[j][t], this.outputActFunct) 
					* this.bwoutput[j][t];
		}
		// cell output & output gate error signals
		for(int i=0; i<this.hiddenSize; i++) {
			double bwioutgate = 0;
			for(int c=0; c<this.cellSize; c++) {
				double bwic=0;
				for(int j=0; j<this.outputSize; j++) {
					bwic += this.weightsOutput[i][c][j] * this.deltaoutput[j][t];
				}
				if(nextt >= 0) { // recurrence within hidden layers
					for(int h=0; h<this.hiddenSize; h++) {
						for(int cto=0; cto<this.cellSize; cto++) { // recurrent cell output to cell input
							bwic += this.weightsCellHidden[i][c][h][cto] * this.deltainp[h][cto][nextt];
						}
						for(int g=0; g<3; g++) { // recurrence cell output to CEC gate input
							bwic += this.weightsGatesHidden[g][i][c][h] * this.deltagates[g][h][nextt];
						}
					}
				}
				this.bwoutng[i][c][t] = bwic;
				bwioutgate += bwic * this.actoutng[i][c][t];
				this.bwoutgated[i][c][t] = bwic * this.actgates[this.outputGateIndex][i][t];
				this.deltaout[i][c][t] = ActivationFunctions.tanhDx(this.netout[i][c][t]) * this.bwoutgated[i][c][t];
			}
			// now the corresponding output gate bw and delta determination... 
			this.bwgates[this.outputGateIndex][i][t] = bwioutgate;
			this.deltagates[this.outputGateIndex][i][t] = ActivationFunctions.sigmoidDx(this.netgates[this.outputGateIndex][i][t]) 
					* bwioutgate;
		}
		// cell state error propagation back until cell input connection
		for(int i=0; i<this.hiddenSize; i++) {
			double bwfgit = 0;
			double bwigit = 0;
			for(int c=0; c<this.cellSize; c++) {
				double zeta = this.deltaout[i][c][t] + 
						this.weightsPeepholes[i][c][this.outputGateIndex] * this.deltagates[this.outputGateIndex][i][t];
				if(nextt > 0) { // recurrence within hidden layers
					zeta += this.actgates[this.forgetGateIndex][i][nextt] * this.deltacec[i][c][nextt] +
							this.weightsPeepholes[i][c][this.inputGateIndex] * this.deltagates[this.inputGateIndex][i][nextt] +
							this.weightsPeepholes[i][c][this.forgetGateIndex] * this.deltagates[this.forgetGateIndex][i][nextt];
				}
				this.deltacec[i][c][t] = this.bwcec[i][c][t] = zeta;
				this.bwinpng[i][c][t] = zeta;
				this.bwinpgated[i][c][t] = this.actgates[this.inputGateIndex][i][t] * zeta;
				this.deltainp[i][c][t] = ActivationFunctions.tanhDx(this.netinp[i][c][t]) * this.bwinpgated[i][c][t];
				if(prevt >= 0) {
					bwfgit += this.actcec[i][c][prevt] * zeta;
				}
				bwigit += this.actinpng[i][c][t] * zeta; 
			}
			// forget gate bw and delta
			this.bwgates[this.forgetGateIndex][i][t] = bwfgit;
			this.deltagates[this.forgetGateIndex][i][t] = ActivationFunctions.sigmoidDx(this.netgates[this.forgetGateIndex][i][t]) * bwfgit;
			// input gate bw and delta
			this.bwgates[this.inputGateIndex][i][t] = bwigit;
			this.deltagates[this.inputGateIndex][i][t] = ActivationFunctions.sigmoidDx(this.netgates[this.inputGateIndex][i][t]) * bwigit;
		}
		// cell input bws ... 
		for(int i=0; i<this.inputSize; i++) {
			double bwi = 0;
			for(int h=0; h<this.hiddenSize; h++) {
				for(int cto=0; cto<this.cellSize; cto++) {
					bwi += this.weightsCellInput[i][h][cto] * this.deltainp[h][cto][t];
				}
				for(int g=0; g<3; g++) {
					bwi += this.weightsGatesInput[g][i][h] * this.deltagates[g][h][t];
				}
			}
			this.bwinput[i][t] = bwi;
		}
	}


	public void backwardPassZero(final double[][] target) {
		//
		final int steps = this.lastinputlength;
		int t_target = target.length - 1; // can be used to separate target length from sequence length... (e.g. if only the last n targets are defined).
		// compute reversely in time.
		for (int t = steps; t >= 0; t--,t_target--) {
			if (t_target >= 0) {
				for (int j = 0; j < this.outputSize; j++) {
					this.bwoutput[j][t] = this.actoutput[j][t] - target[t_target][j];
				}
				if(t == steps)
					this.oneBackwardPassAt(t, t-1, -1);
				else
					this.oneBackwardPassAt(t, t-1, t+1);
			}else{ // target length is used up... zero error coming from all other outputs
				for (int j = 0; j < this.outputSize; j++) {
					this.bwoutput[j][t] = 0;
				}
				this.oneBackwardPassAt(t, t-1, t+1);
			}
		}
		// ################################################
		// Compute the resulting weight derivatives.
		// ################################################
		setWeightsDerivativesGeneral(0, steps);
	}

	/**
	 * This is the general method that can process also weight derivatives where tInit>tLast,
	 * that is, when the indices of the last time sequence online went over the buffer (buffer is circular)
	 * @param tInit
	 * @param tLast
	 */
	public void setWeightsDerivativesGeneral(int tInit, int tLast) {
		if(tInit <= tLast) {
			setWeightsDerivatives(tInit, tLast);
			return;
		}
		// circular buffer case... 
		for(int g=0; g<3; g++) {
			// ff: private double[][][] dweightsGatesInput; // [to gate ig/hg/og][from-index(inputSize)][to-index(hiddenSize)]
			for(int i=0; i<this.inputSize; i++) {
				for(int j=0; j<this.hiddenSize; j++) {
					double dw = 0;
					for(int t=tInit; t<this.bufferlength; t++) {
						dw += this.actinput[i][t] * this.deltagates[g][j][t];
					}
					for(int t=0; t<tLast+1; t++) {
						dw += this.actinput[i][t] * this.deltagates[g][j][t];
					}
					this.dweightsGatesInput[g][i][j] = dw;
				}
			}
			if(this.usebias[0]) {
				for(int j=0; j<this.hiddenSize; j++) {
					double dw = 0;
					for(int t=tInit; t<this.bufferlength; t++) {
						dw += ANNLayer_LSTM.BIAS * this.deltagates[g][j][t];
					}
					for(int t=0; t<tLast+1; t++) {
						dw += ANNLayer_LSTM.BIAS * this.deltagates[g][j][t];
					}
					this.dweightsGatesInput[g][this.inputSize][j] = dw;
				}				
			}
			// fb: private double[][][][] dweightsGatesHidden; // [to gate ig/hg/og][from-cell-index(hiddenSize)][from-CEC-cell-index][to-index(hiddenSize)]
			for(int i=0; i<this.hiddenSize; i++) {
				for(int c=0; c<this.cellSize; c++) {
					for(int j=0; j<this.hiddenSize; j++) {
						double dw = 0;
						for(int t=tInit; t<this.bufferlength-1; t++) {
							dw += this.actoutgated[i][c][t] * this.deltagates[g][j][t+1];
						}
						dw += this.actoutgated[i][c][this.bufferlength-1] * this.deltagates[g][j][0];
						for(int t=0; t<tLast; t++) {
							dw += this.actoutgated[i][c][t] * this.deltagates[g][j][t+1];
						}
						this.dweightsGatesHidden[g][i][c][j] = dw;
					}
				}
			}
		}
		// ff: private double[][][] dweightsCellInput; // [from-index(inputSize)][to-cell-index(hiddenSize)][to-CEC-cell-index(cellSize)]
		for(int i=0; i<this.inputSize; i++) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int c=0; c<this.cellSize; c++) {
					double dw = 0;
					for(int t=tInit; t<this.bufferlength; t++) {
						dw += this.actinput[i][t] *  this.deltainp[j][c][t];
					}
					for(int t=0; t<tLast+1; t++) {
						dw += this.actinput[i][t] *  this.deltainp[j][c][t];
					}
					this.dweightsCellInput[i][j][c] = dw;
				}
			}
		}
		if(this.usebias[1]) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int c=0; c<this.cellSize; c++) {
					double dw = 0;
					for(int t=tInit; t<this.bufferlength; t++) {
						dw += ANNLayer_LSTM.BIAS *  this.deltainp[j][c][t];
					}
					for(int t=0; t<tLast+1; t++) {
						dw += ANNLayer_LSTM.BIAS *  this.deltainp[j][c][t];
					}
					this.dweightsCellInput[this.inputSize][j][c] = dw;
				}
			}			
		}
		// fb: private double[][][][] weightsCellHidden; // [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][to-cell-index(hiddenSize)][to-CEC-cell-index(cellSize)]
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.hiddenSize; j++) {
					for(int cto=0; cto<this.cellSize; cto++) {
						double dw = 0;
						for(int t=tInit; t<this.bufferlength-1; t++) {
							dw += this.actoutgated[i][cfrom][t] *  this.deltainp[j][cto][t+1];
						}
						dw += this.actoutgated[i][cfrom][this.bufferlength-1] *  this.deltainp[j][cto][0];
						for(int t=0; t<tLast; t++) {
							dw += this.actoutgated[i][cfrom][t] *  this.deltainp[j][cto][t+1];
						}
						this.dweightsCellHidden[i][cfrom][j][cto] = dw;
					}
				}
			}
		}
		// private double[][][] weightsPeepholes; // [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][ig/hg/og] 
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				double dwig = 0, dwfg = 0, dwog = 0;
				for(int t=tInit; t<this.bufferlength-1; t++) {
					dwig += this.actcec[i][cfrom][t] * this.deltagates[this.inputGateIndex][i][t+1];
					dwfg += this.actcec[i][cfrom][t] * this.deltagates[this.forgetGateIndex][i][t+1];
					dwog += this.actcec[i][cfrom][t] * this.deltagates[this.outputGateIndex][i][t];
				}
				dwig += this.actcec[i][cfrom][this.bufferlength-1] * this.deltagates[this.inputGateIndex][i][0];
				dwfg += this.actcec[i][cfrom][this.bufferlength-1] * this.deltagates[this.forgetGateIndex][i][0];
				dwog += this.actcec[i][cfrom][this.bufferlength-1] * this.deltagates[this.outputGateIndex][i][this.bufferlength-1];
				for(int t=0; t<tLast; t++) {
					dwig += this.actcec[i][cfrom][t] * this.deltagates[this.inputGateIndex][i][t+1];
					dwfg += this.actcec[i][cfrom][t] * this.deltagates[this.forgetGateIndex][i][t+1];
					dwog += this.actcec[i][cfrom][t] * this.deltagates[this.outputGateIndex][i][t];
				}
				dwog += this.actcec[i][cfrom][tLast] * this.deltagates[this.outputGateIndex][i][tLast];
				// assigning the values.
				this.dweightsPeepholes[i][cfrom][this.inputGateIndex] = dwig;
				this.dweightsPeepholes[i][cfrom][this.forgetGateIndex] = dwfg;
				this.dweightsPeepholes[i][cfrom][this.outputGateIndex] = dwog;
			}
		}		
		// private double[][][] weightsOutput; //  [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][to-index(outputSize)]
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.outputSize; j++) {
					double dw = 0;
					for(int t=tInit; t<this.bufferlength; t++) {
						dw += this.actoutgated[i][cfrom][t] * this.deltaoutput[j][t];
					}
					for(int t=0; t<tLast+1; t++) {
						dw += this.actoutgated[i][cfrom][t] * this.deltaoutput[j][t];
					}
					this.dweightsOutput[i][cfrom][j] = dw;
				}
			}
		}
		if(this.usebias[2]) {
			for(int j=0; j<this.outputSize; j++) {
				double dw = 0;
				for(int t=tInit; t<this.bufferlength; t++) {
					dw += ANNLayer_LSTM.BIAS * this.deltaoutput[j][t];
				}
				for(int t=0; t<tLast+1; t++) {
					dw += ANNLayer_LSTM.BIAS * this.deltaoutput[j][t];
				}
				this.dweightsOutput[this.hiddenSize][0][j] = dw;
			}			
		}
	}

	private void setWeightsDerivatives(int tInit, int tLast) {
		for(int g=0; g<3; g++) {
			// ff: private double[][][] dweightsGatesInput; // [to gate ig/hg/og][from-index(inputSize)][to-index(hiddenSize)]
			for(int i=0; i<this.inputSize; i++) {
				for(int j=0; j<this.hiddenSize; j++) {
					double dw = 0;
					for(int t=0; t<tLast+1; t++) {
						dw += this.actinput[i][t] * this.deltagates[g][j][t];
					}
					this.dweightsGatesInput[g][i][j] = dw;
				}
			}
			if(this.usebias[0]) {
				for(int j=0; j<this.hiddenSize; j++) {
					double dw = 0;
					for(int t=0; t<tLast+1; t++) {
						dw += ANNLayer_LSTM.BIAS * this.deltagates[g][j][t];
					}
					this.dweightsGatesInput[g][this.inputSize][j] = dw;
				}				
			}
			// fb: private double[][][][] dweightsGatesHidden; // [to gate ig/hg/og][from-cell-index(hiddenSize)][from-CEC-cell-index][to-index(hiddenSize)]
			for(int i=0; i<this.hiddenSize; i++) {
				for(int c=0; c<this.cellSize; c++) {
					for(int j=0; j<this.hiddenSize; j++) {
						double dw = 0;
						for(int t=0; t<tLast; t++) {
							dw += this.actoutgated[i][c][t] * this.deltagates[g][j][t+1];
						}
						this.dweightsGatesHidden[g][i][c][j] = dw;
					}
				}
			}
		}
		// ff: private double[][][] dweightsCellInput; // [from-index(inputSize)][to-cell-index(hiddenSize)][to-CEC-cell-index(cellSize)]
		for(int i=0; i<this.inputSize; i++) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int c=0; c<this.cellSize; c++) {
					double dw = 0;
					for(int t=0; t<tLast+1; t++) {
						dw += this.actinput[i][t] *  this.deltainp[j][c][t];
					}
					this.dweightsCellInput[i][j][c] = dw;
				}
			}
		}
		if(this.usebias[1]) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int c=0; c<this.cellSize; c++) {
					double dw = 0;
					for(int t=0; t<tLast+1; t++) {
						dw += ANNLayer_LSTM.BIAS *  this.deltainp[j][c][t];
					}
					this.dweightsCellInput[this.inputSize][j][c] = dw;
				}
			}			
		}
		// fb: private double[][][][] weightsCellHidden; // [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][to-cell-index(hiddenSize)][to-CEC-cell-index(cellSize)]
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.hiddenSize; j++) {
					for(int cto=0; cto<this.cellSize; cto++) {
						double dw = 0;
						for(int t=0; t<tLast; t++) {
							dw += this.actoutgated[i][cfrom][t] *  this.deltainp[j][cto][t+1];
						}
						this.dweightsCellHidden[i][cfrom][j][cto] = dw;
					}
				}
			}
		}
		// private double[][][] weightsPeepholes; // [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][ig/hg/og] 
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				double dwig = 0, dwfg = 0, dwog = 0;
				for(int t=0; t<tLast; t++) {
					dwig += this.actcec[i][cfrom][t] * this.deltagates[this.inputGateIndex][i][t+1];
					dwfg += this.actcec[i][cfrom][t] * this.deltagates[this.forgetGateIndex][i][t+1];
					dwog += this.actcec[i][cfrom][t] * this.deltagates[this.outputGateIndex][i][t];
				}
				dwog += this.actcec[i][cfrom][tLast] * this.deltagates[this.outputGateIndex][i][tLast]; // ff connection still get the final error signal
				// assigning the values.
				this.dweightsPeepholes[i][cfrom][this.inputGateIndex] = dwig;
				this.dweightsPeepholes[i][cfrom][this.forgetGateIndex] = dwfg;
				this.dweightsPeepholes[i][cfrom][this.outputGateIndex] = dwog;
			}
		}		
		// private double[][][] weightsOutput; //  [from-cell-index(hiddenSize)][from-CEC-cell-index(cellSize)][to-index(outputSize)]
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.outputSize; j++) {
					double dw = 0;
					for(int t=0; t<tLast+1; t++) {
						dw += this.actoutgated[i][cfrom][t] * this.deltaoutput[j][t];
					}
					this.dweightsOutput[i][cfrom][j] = dw;
				}
			}
		}
		if(this.usebias[2]) {
			for(int j=0; j<this.outputSize; j++) {
				double dw = 0;
				for(int t=0; t<tLast+1; t++) {
					dw += ANNLayer_LSTM.BIAS * this.deltaoutput[j][t];
				}
				this.dweightsOutput[this.hiddenSize][0][j] = dw;
			}			
		}
	}


	/**
	 * This method adapts the internal hidden state activity to tune the sequence
	 * towards generating the desired target output after depth steps. 
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
			double momentumRate,
			ErrorBasedLearner.WEIGHT_UPDATE activeInferenceUpdateType
			) {
		//
		this.rebufferOnDemand(depth+1);
		// input is empty... I want to infer the hidden states. 
		double[][] zeroInput = new double[depth][this.inputSize];
		final int numHiddenActivities = this.getHiddenActivitiesNum();

		ErrorBasedLearner ebl = null;
		switch(activeInferenceUpdateType) {
		case GradientDescent:
			ebl = new GradientDescentLearner(numHiddenActivities, learningRate, momentumRate);
			break;
		case Adam:
			ebl = new AdamLearner(numHiddenActivities, learningRate);
			break;
        default:
            break;
		}
		double initialActivities[] = new double[numHiddenActivities];
		ebl.resetValues(initialActivities);		
		double bwErrors[] = new double[numHiddenActivities];

		// begin of adaptation process
		for(int epoch=0; epoch<numEpochs; epoch++) {
			// note: internal activity of last epoch is intentionally used here as the internal start state
			this.forwardPassZero(zeroInput); 
//			double[] output = getOutputActivities(this.lastinputlength);
//			System.out.println("Current error (RMSE): "+Tools.RMSE(output, target));
			this.backwardPassZero(new double[][]{target});
			// now adapt the initial internal, hidden activity of the network!
			this.readBWErrors(bwErrors, 0);
			double newActivities[] = ebl.learningIteration(bwErrors);
			this.writeInternalActivities(newActivities, 0);
		}
		// done... thus moving the hidden state of activation towards the aimed-at state.
		for(int t=0; t<depth; t++)
			this.forwardPass(zeroInput[0]);	// moving the hidden activations towards the aimed-at start activations.	
		double[] output = getOutputActivities(0);
		System.out.println("Active Inference final target error (RMSE): "+Tools.RMSE(output, target));
	}

	@Override
	public int getHiddenActivitiesNum() {
		return this.hiddenSize + this.hiddenSize * this.cellSize; // forget gate state and cec cells states are sufficient!!! 
	}
	
	@Override
	public void getInternalActivityBounds(double[] lowerBounds,  double[] upperBounds) {
		int idx = 0;
		for(int i=0; i<this.hiddenSize; i++) {
			lowerBounds[idx] = ActivationFunctions.getLowerBound(ACT_FUNCT.Sigmoid);
			upperBounds[idx] = ActivationFunctions.getUpperBound(ACT_FUNCT.Sigmoid);
			idx++;
			for(int c=0; c<this.cellSize; c++) {
				lowerBounds[idx] = ActivationFunctions.getLowerBound(ACT_FUNCT.Linear);
				upperBounds[idx] = ActivationFunctions.getUpperBound(ACT_FUNCT.Linear);
				idx++;
			}
		}
		
	}

	@Override
	public void readInternalActivities(double[] activities, int timePoint) {
		int idx=0;
		for(int i=0; i<this.hiddenSize; i++) {
			activities[idx++] = this.actgates[this.outputGateIndex][i][timePoint];
			for(int c=0; c<this.cellSize; c++) {
				activities[idx++] = this.actcec[i][c][timePoint];
			}
		}
	}

	@Override
	public void readBWErrors(double[] bwErrors, int timePoint) {
		int idx=0;
		for(int i=0; i<this.hiddenSize; i++) {
			bwErrors[idx] = this.bwgates[this.outputGateIndex][i][timePoint];
			idx++;
			for(int c=0; c<this.cellSize; c++) {
				bwErrors[idx] = this.bwcec[i][c][timePoint];
				idx++;				
			}
		}
	}

	@Override
	public void writeInternalActivities(double[] activities, int timePoint) {
		int idx=0;
		for(int i=0; i<this.hiddenSize; i++) {
			this.actgates[this.outputGateIndex][i][timePoint] = activities[idx++];
			for(int c=0; c<this.cellSize; c++) {
				this.actcec[i][c][timePoint] = activities[idx++];
			}
		}
		// NOTE:... am propagating the cec activities to the output still ... such that the recurrence is properly initialized.
		for(int i=0; i<this.hiddenSize; i++) {
			for(int c=0; c<this.cellSize; c++) {
				this.netout[i][c][timePoint] = actcec[i][c][timePoint];
				this.actoutng[i][c][timePoint] = ActivationFunctions.tanh(netout[i][c][timePoint]);
				this.actoutgated[i][c][timePoint] = this.actoutng[i][c][timePoint] * this.actgates[this.outputGateIndex][i][timePoint];
			}
		}
	}


	/**
	 * Initializes the weights randomly and normal distributed with
	 * std. dev. 0.1.
	 * @param rnd Instance of Random.
	 */
	public void initializeWeights(final Random rnd, final double stddev) {
		double[] weights = new double[this.numWeights];
		for(int i=0; i<this.numWeights; i++) 
			weights[i] = rnd.nextGaussian() * stddev;
		writeWeights(weights);
	}

	public int getNumWeights() {
		return this.numWeights;
	}

	@Override
	public int getOutputSize() {
		return this.outputSize;
	}

	@Override
	public int getInputSize() {
		return this.inputSize;
	}

	/**
	 * Writing the specified weight values to the individual weights in this LSTM structure. 
	 * 
	 * @param weights array of weights - should be of size weightsnum
	 */
	public void writeWeights(final double[] weights) {
		assert(weights.length == this.numWeights);

		int idx = 0; 
		for(int g=0; g<3; g++) {
			for(int i=0; i<this.inputSize; i++) {
				for(int j=0; j<this.hiddenSize; j++) {
					this.weightsGatesInput[g][i][j] = weights[idx];
					idx++;
				}
			}
			if(this.usebias[0]) {
				for(int j=0; j<this.hiddenSize; j++) {
					this.weightsGatesInput[g][this.inputSize][j] = weights[idx];
					idx++;
				}
			}
			for(int i=0; i<this.hiddenSize; i++) {
				for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
					for(int j=0; j<this.hiddenSize; j++) {
						this.weightsGatesHidden[g][i][cfrom][j] = weights[idx];
						idx++;
					}
				}
			}
		}
		for(int i=0; i<this.inputSize; i++) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int cto=0; cto<this.cellSize; cto++) {
					this.weightsCellInput[i][j][cto] = weights[idx];
					idx++;
				}
			}
		}
		if(this.usebias[1]) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int cto=0; cto<this.cellSize; cto++) {
					this.weightsCellInput[inputSize][j][cto] = weights[idx];
					idx++;
				}
			}			
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.hiddenSize; j++) {
					for(int cto=0; cto<this.cellSize; cto++) {
						this.weightsCellHidden[i][cfrom][j][cto] = weights[idx];
						idx++;
					}	
				}
			}
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int g=0; g<3; g++) {
					this.weightsPeepholes[i][cfrom][g] = weights[idx];
					idx++;
				}
			}
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.outputSize; j++) {
					this.weightsOutput[i][cfrom][j] = weights[idx];
					idx++;
				}
			}
		}		
		if(this.usebias[2]) {
			for(int j=0; j<this.outputSize; j++) {
				this.weightsOutput[this.hiddenSize][0][j] = weights[idx];
				idx++;
			}			
		}
		assert(idx==weights.length);
	}

	/**
	 * Reading the weights in the network in the provided weights vector.
	 * 
	 * @param weights the array into which the current weights of the network will be written to.
	 */
	public void readWeights(final double[] weights) {
		assert(weights.length == this.numWeights);

		int idx = 0; 
		for(int g=0; g<3; g++) {
			for(int i=0; i<this.inputSize; i++) {
				for(int j=0; j<this.hiddenSize; j++) {
					weights[idx] = this.weightsGatesInput[g][i][j];
					idx++;
				}
			}
			if(this.usebias[0]) {
				for(int j=0; j<this.hiddenSize; j++) {
					weights[idx] = this.weightsGatesInput[g][this.inputSize][j];
					idx++;
				}
			}
			for(int i=0; i<this.hiddenSize; i++) {
				for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
					for(int j=0; j<this.hiddenSize; j++) {
						weights[idx] = this.weightsGatesHidden[g][i][cfrom][j];
						idx++;
					}
				}
			}
		}
		for(int i=0; i<this.inputSize; i++) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int cto=0; cto<this.cellSize; cto++) {
					weights[idx] = this.weightsCellInput[i][j][cto];
					idx++;
				}
			}
		}
		if(this.usebias[1]) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int cto=0; cto<this.cellSize; cto++) {
					weights[idx] = this.weightsCellInput[inputSize][j][cto];
					idx++;
				}
			}			
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.hiddenSize; j++) {
					for(int cto=0; cto<this.cellSize; cto++) {
						weights[idx] = this.weightsCellHidden[i][cfrom][j][cto];
						idx++;
					}	
				}
			}
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int g=0; g<3; g++) {
					weights[idx] = this.weightsPeepholes[i][cfrom][g];
					idx++;
				}
			}
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.outputSize; j++) {
					weights[idx] = this.weightsOutput[i][cfrom][j];
					idx++;
				}
			}
		}
		if(this.usebias[2]) {
			for(int j=0; j<this.outputSize; j++) {
				weights[idx] = this.weightsOutput[this.hiddenSize][0][j];
				idx++;
			}			
		}
		assert(idx==weights.length);
	}

	/**
	 * Reading the weights in the network in the provided weights vector.
	 * 
	 * @param weights the array into which the current weights of the network will be written to.
	 */
	public void readDiffWeights(final double[] myDiffWeights) {
		assert(myDiffWeights.length == this.numWeights);

		int idx = 0; 
		for(int g=0; g<3; g++) {
			for(int i=0; i<this.inputSize; i++) {
				for(int j=0; j<this.hiddenSize; j++) {
					myDiffWeights[idx] = this.dweightsGatesInput[g][i][j];
					idx++;
				}
			}
			if(this.usebias[0]) {
				for(int j=0; j<this.hiddenSize; j++) {
					myDiffWeights[idx] = this.dweightsGatesInput[g][this.inputSize][j];
					idx++;
				}
			}
			for(int i=0; i<this.hiddenSize; i++) {
				for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
					for(int j=0; j<this.hiddenSize; j++) {
						myDiffWeights[idx] = this.dweightsGatesHidden[g][i][cfrom][j];
						idx++;
					}
				}
			}
		}
		for(int i=0; i<this.inputSize; i++) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int cto=0; cto<this.cellSize; cto++) {
					myDiffWeights[idx] = this.dweightsCellInput[i][j][cto];
					idx++;
				}
			}
		}
		if(this.usebias[1]) {
			for(int j=0; j<this.hiddenSize; j++) {
				for(int cto=0; cto<this.cellSize; cto++) {
					myDiffWeights[idx] = this.dweightsCellInput[inputSize][j][cto];
					idx++;
				}
			}			
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.hiddenSize; j++) {
					for(int cto=0; cto<this.cellSize; cto++) {
						myDiffWeights[idx] = this.dweightsCellHidden[i][cfrom][j][cto];
						idx++;
					}	
				}
			}
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int g=0; g<3; g++) {
					myDiffWeights[idx] = this.dweightsPeepholes[i][cfrom][g];
					idx++;
				}
			}
		}
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				for(int j=0; j<this.outputSize; j++) {
					myDiffWeights[idx] = this.dweightsOutput[i][cfrom][j];
					idx++;
				}
			}
		}
		if(this.usebias[2]) {
			for(int j=0; j<this.outputSize; j++) {
				myDiffWeights[idx] = this.dweightsOutput[this.hiddenSize][0][j];
				idx++;
			}			
		}
	}


	/**
	 * Reading the weights in the network in the provided weights vector.
	 * 
	 * @param weights the array into which the current weights of the network will be written to.
	 */
	public String toStringWeights() {
		StringBuffer sb = new StringBuffer();
		for(int g=0; g<3; g++) {
			sb.append("Gate "+g+" input:\n");
			for(int i=0; i<this.inputSize; i++) {
				sb.append("I"+i+" to: ");
				sb.append(Tools.toStringArray(this.weightsGatesInput[g][i]));
				sb.append("\n");
			}
			if(this.usebias[0]) {
				sb.append("BiasW: ");
				sb.append(Tools.toStringArray(this.weightsGatesInput[g][this.inputSize]));
				sb.append("\n");
			}
			sb.append("Gate "+g+" recurrences:\n");
			for(int i=0; i<this.hiddenSize; i++) {
				for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
					sb.append("H"+i+","+cfrom+" to: ");
					sb.append(Tools.toStringArray(this.weightsGatesHidden[g][i][cfrom]));
					sb.append("\n");
				}
			}
		}
		sb.append("Cell Input:\n");
		for(int i=0; i<this.inputSize; i++) {
			sb.append("I"+i+" to: ");
			sb.append(Tools.toStringArray(this.weightsCellInput[i]));
			sb.append("\n");
		}
		if(this.usebias[1]) {
			sb.append("BiasCI: ");
			sb.append(Tools.toStringArray(this.weightsCellInput[inputSize]));
			sb.append("\n");
		}
		sb.append("Cell Input Recurrences:\n");
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				sb.append("H"+i+","+cfrom+" to: ");
				sb.append(Tools.toStringArray(this.weightsCellHidden[i][cfrom]));
				sb.append("\n");
			}
		}
		sb.append("Peepholes:\n");
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				sb.append("H"+i+","+cfrom+" to: ");
				sb.append(Tools.toStringArray(this.weightsPeepholes[i][cfrom]));
				sb.append("\n");
			}
		}
		sb.append("Cell Output:\n");
		for(int i=0; i<this.hiddenSize; i++) {
			for(int cfrom=0; cfrom<this.cellSize; cfrom++) {
				sb.append("H"+i+","+cfrom+" to: ");
				sb.append(Tools.toStringArray(this.weightsOutput[i][cfrom]));
				sb.append("\n");
			}
		}
		if(this.usebias[2]) {
			sb.append("BiasCI: ");
			sb.append(Tools.toStringArray(this.weightsOutput[this.hiddenSize][0]));
			sb.append("\n");
		}
		return sb.toString();
	}

	public String toStringActivities() {
		StringBuffer sb = new StringBuffer();
		sb.append("Input:\n");
		sb.append(Tools.toStringArray(this.actinput,this.actinput.length, this.lastinputlength+1));
		sb.append("\n");
		//		
		sb.append("Hidden:\n");
		for(int i=0; i<this.hiddenSize; i++) {
			sb.append("H+G+Out"+(i+1)+":\n");
			for(int c=0; c<this.cellSize; c++) {
				sb.append(Tools.toStringArray(this.actcec[i][c],this.lastinputlength+1));
			}
			sb.append("\n");
			for(int g=0; g<3; g++) {
				sb.append(Tools.toStringArray(this.actgates[g][i],this.lastinputlength+1));
			}
			sb.append("\n");
			for(int c=0; c<this.cellSize; c++) {
				sb.append(Tools.toStringArray(this.actoutgated[i][c],this.lastinputlength+1));
			}
			sb.append("\n");
		}
		//		
		sb.append("Output:\n");
		sb.append(Tools.toStringArray(this.actoutput,this.actoutput.length,this.lastinputlength+1));
		return sb.toString();
	}


	public String toStringNetInputs() {
		StringBuffer sb = new StringBuffer();
		sb.append("Hidden:\n");
		for(int i=0; i<this.hiddenSize; i++) {
			sb.append("H+G+Out"+(i+1)+":\n");
			for(int c=0; c<this.cellSize; c++) {
				sb.append(Tools.toStringArray(this.netcec[i][c],this.lastinputlength+1));
			}
			sb.append("\n");
			for(int g=0; g<3; g++) {
				sb.append(Tools.toStringArray(this.netgates[g][i],this.lastinputlength+1));
			}
			sb.append("\n");
			for(int c=0; c<this.cellSize; c++) {
				sb.append(Tools.toStringArray(this.netout[i][c],this.lastinputlength+1));
			}
			sb.append("\n");
		}
		//		
		sb.append("Output:\n");
		sb.append(Tools.toStringArray(this.netoutput,this.netoutput.length,this.lastinputlength+1));
		return sb.toString();
	}

	public String toStringBWError() {
		StringBuffer sb = new StringBuffer();
		sb.append("Input:\n");
		sb.append(Tools.toStringArray(this.bwinput,this.bwinput.length, this.lastinputlength+1));
		sb.append("\n");
		//		
		sb.append("Hidden:\n");
		for(int i=0; i<this.hiddenSize; i++) {
			sb.append("H+G+Out"+(i+1)+":\n");
			for(int c=0; c<this.cellSize; c++) {
				sb.append(Tools.toStringArray(this.bwcec[i][c],this.lastinputlength+1));
			}
			sb.append("\n");
			for(int g=0; g<3; g++) {
				sb.append(Tools.toStringArray(this.bwgates[g][i],this.lastinputlength+1));
			}
			sb.append("\n");
			for(int c=0; c<this.cellSize; c++) {
				sb.append(Tools.toStringArray(this.bwoutgated[i][c],this.lastinputlength+1));
			}
			sb.append("\n");
		}
		//		
		sb.append("Output:\n");
		sb.append(Tools.toStringArray(this.bwoutput,this.bwoutput.length,this.lastinputlength+1));
		return sb.toString();
	}


	public String toStringDeltas() {
		StringBuffer sb = new StringBuffer();
		sb.append("Hidden:\n");
		for(int i=0; i<this.hiddenSize; i++) {
			sb.append("H+G+Out"+(i+1)+":\n");
			for(int c=0; c<this.cellSize; c++) {
				sb.append(Tools.toStringArray(this.deltacec[i][c],this.lastinputlength+1));
			}
			sb.append("\n");
			for(int g=0; g<3; g++) {
				sb.append(Tools.toStringArray(this.deltagates[g][i],this.lastinputlength+1));
			}
			sb.append("\n");
			for(int c=0; c<this.cellSize; c++) {
				sb.append(Tools.toStringArray(this.deltaout[i][c],this.lastinputlength+1));
			}
			sb.append("\n");
		}
		//		
		sb.append("Output:\n");
		sb.append(Tools.toStringArray(this.deltaoutput,this.deltaoutput.length,this.lastinputlength+1));
		return sb.toString();
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
		final double[] dweights      = new double[this.numWeights];

		ErrorBasedLearner ebl = null;
		switch(weightUpdateType) {
		case GradientDescent:
			ebl = new GradientDescentLearner(this.numWeights, learningRate, momentumRate);
			break;
		case Adam:
			ebl = new AdamLearner(this.numWeights, learningRate);
			break;
        default:
            break;
		}
		this.readWeights(initialWeights);
		ebl.resetValues(initialWeights);

		// create initial index permutation.
		final int[] indices = new int[input.length];
		for (int i = 0; i < indices.length; i++) {
			indices[i] = i;
		}
		double error = Double.POSITIVE_INFINITY;

		// epoch loop.
		for (int i = 0; i < epochs; i++) {
			// shuffle indices.
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
				//				if(i+1 == epochs) {
				//					System.out.println("Net Inputs: ###############################");
				//					System.out.println(this.toStringNetInputs());
				//					System.out.println("Activities: ###############################");
				//					System.out.println(this.toStringActivities());
				//					System.out.println("BW Errors: ###############################");
				//					System.out.println(this.toStringBWError());
				//					System.out.println("Deltas: ###############################");
				//					System.out.println(this.toStringDeltas());
				//					System.out.println("Target vs. Output: ###############################");
				//					System.out.println(""+Tools.toStringArray(input[indices[j]])+"\n o->"+Tools.toStringArray(output)+
				//							"\n t->"+Tools.toStringArray(target[indices[j]]));
				//				}
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
	@Deprecated
	public double trainStochasticEpochWise(
			final Random rnd, 
			final double[][][] input, 
			final double target[][][],
			final double epochs,
			final double learningrate,
			final double momentumrate,
			final LearningListener listener,
			final WEIGHT_UPDATE weightUpdateType
			) {
		//
		assert(input.length == target.length);
		//
		final double[] weights       = new double[this.numWeights];
		final double[] dweights      = new double[this.numWeights];
		double[] weightsupdate = new double[this.numWeights];
		double[] weightsupdatelast = new double[this.numWeights];
		double[] nuvalues = new double[this.numWeights];
		double[] nuvalueslast = new double[this.numWeights];
		//
		this.readWeights(weights);
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
			double beta1 = 0.9;
			double beta1n1 = 0.1;
			double beta2 =  0.999;
			double beta2n1 = 0.001;			
			// train all samples in online manner, i.e. iterate over all samples
			// while considering the shuffled order and update the weights 
			// immediately after each sample
			//
			for(int j=0; j<indices.length; j++) {
				if(i+1 == epochs && j+1 == indices.length) {
					System.out.println("Very last training instance.");
				}
				//				this.resetFirstTimeStepActivitiesToZero(); // when working with forwardPassZero, this is not necessary!
				double[][] output = this.forwardPassZero(input[indices[j]]);
				errorsum += Tools.MSE(output,  target[indices[j]]);				
				if(i+1 == epochs) {
					System.out.println(""+Tools.toStringArray(input[indices[j]])+"\n o->"+Tools.toStringArray(output)+
							"\n t->"+Tools.toStringArray(target[indices[j]]));
				}
				this.backwardPassZero(target[indices[j]]);

				this.readDiffWeights(dweights);

				switch(weightUpdateType) {
				case GradientDescent:
					for(int k=0; k<weights.length; k++) {
						weightsupdate[k] += learningrate * dweights[k];
					}
					break;
				case Adam:
					for(int k=0; k<weights.length; k++) {
						nuvalues[k] += dweights[k] * dweights[k];
						weightsupdate[k] += dweights[k];
					}
					break;
                default:
                    break;

				}
			}

			switch(weightUpdateType) {
			case GradientDescent:
				for(int k=0; k<weights.length; k++) {
					weightsupdate[k] = weightsupdate[k] / indices.length + momentumrate * weightsupdatelast[k];
					weights[k] -= weightsupdate[k];
				}
				break;
			case Adam:
				double tau = i+1;
				double ombeta1tau = 1. - Math.pow(beta1, tau);
				double ombeta2tau = 1. - Math.pow(beta2, tau);
				for(int k=0; k<weights.length; k++) {
					nuvalues[k] = beta2 * nuvalueslast[k] + beta2n1 * nuvalues[k] / indices.length;
					weightsupdate[k] = beta1 * weightsupdatelast[k] + beta1n1 * weightsupdate[k] / indices.length;
					weights[k] -= weightsupdate[k] / ombeta1tau * learningrate / Math.sqrt(nuvalues[k] / ombeta2tau + .000000001);
				}
				break;
            default:
                break;
			}			
			this.writeWeights(weights);
			//
			weightsupdatelast = weightsupdate;
			weightsupdate = new double[this.numWeights];
			if(weightUpdateType==ErrorBasedLearner.WEIGHT_UPDATE.Adam) {
				nuvalueslast = nuvalues;
				nuvalues = new double[this.numWeights];
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
	public double[] getOutputActivities(int timePoint) {
		double[] ret = new double[this.outputSize];
		for(int i=0; i<this.outputSize; i++) {
			ret[i] = this.actoutput[i][timePoint];
		}
		return ret;
	}

	@Override
	public void setOutputBW(double[] bwSignals, int timePoint) {
		for(int o=0; o<this.outputSize; o++) {
			this.bwoutput[o][timePoint] = bwSignals[o];
		}
	}

	@Override
	public double[] getInputBW(int timePoint) {
		double[] ret = new double[this.inputSize];
		for(int i=0; i<this.inputSize; i++) {
			ret[i] = this.bwinput[i][timePoint];
		}
		return ret;
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

	public static ANNLayer_LSTM readNetwork(String fileName) throws IOException, ClassNotFoundException, ClassCastException {
		FileInputStream f = new FileInputStream(fileName);
		ObjectInputStream s = new ObjectInputStream(f);
		ANNLayer_LSTM rnn = (ANNLayer_LSTM)s.readObject();
		s.close();
		f.close();
		return rnn;
	}

}
