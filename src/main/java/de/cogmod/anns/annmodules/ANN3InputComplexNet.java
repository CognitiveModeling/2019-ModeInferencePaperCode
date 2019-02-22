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
import de.cogmod.problems.OutputToInputMapping;
import de.cogmod.utilities.LearningListener;
import de.cogmod.utilities.Tools;
import de.cogmod.utilities.ActivationFunctions.ACT_FUNCT;

public class ANN3InputComplexNet implements Serializable, ANNLayer{
	// needed to allow serialization.
	private static final long serialVersionUID = 1L;

	private ANNLayer[] layers;
	private final int numLayers;
	private final int sensorInputSize;
	private final int motorInputSize;
	private final int sensorimotorSize;
	private final int contextInputSize;
	private final int summedInputSize;
	private final int outputSize;

	private int lastSequenceLength = 0;
	private int bufferlength    = 0;
	
	private final double[] zeroTargetValues;
	private final int numWeights;
	private final double[][] myLayerWeights;
	private final double[][] myLayerDiffWeights;

	private final double[][] myHiddenBWErrors;
	private final double[][] myHiddenActivities;
	
	private final int myHiddenNeuronsSize;
	private final double[] myHiddenNeuronsLowerBounds;
	private final double[] myHiddenNeuronsUpperBounds;
	
	public ANN3InputComplexNet(int sensorInputSize, int motorInputSize, int contextInputSize, 
			ANNLayer.LayerTypes[] layTypes, 
			int[][] parLay, 
			boolean[][] biasesLay, 
			ACT_FUNCT[][] actFunctLay) {
		//
		this.numLayers = layTypes.length;
		layers = new ANNLayer[numLayers];
		//
		this.myLayerWeights = new double[this.numLayers][];
		this.myLayerDiffWeights = new double[this.numLayers][];
		this.myHiddenBWErrors = new double[this.numLayers][];
		this.myHiddenActivities = new double[this.numLayers][];
		//
		this.sensorInputSize = sensorInputSize;
		this.motorInputSize = motorInputSize;
		this.sensorimotorSize =  sensorInputSize + motorInputSize;
		this.contextInputSize = contextInputSize;
		this.summedInputSize = sensorInputSize+motorInputSize+contextInputSize;
		int is = this.summedInputSize;
		int numW = 0;
		for(int l=0; l<this.numLayers; l++) {
			this.layers[l] = ANN3InputComplexNet.constructANNLayer(is, layTypes[l], parLay[l], 
					biasesLay[l], actFunctLay[l]);
			is = this.layers[l].getOutputSize();
			int numWeightsInLayer = this.layers[l].getNumWeights();
			numW += numWeightsInLayer;
			this.myLayerWeights[l] = new double[numWeightsInLayer];
			this.myLayerDiffWeights[l] = new double[numWeightsInLayer];
			this.myHiddenBWErrors[l] = new double[this.layers[l].getHiddenActivitiesNum()];
			this.myHiddenActivities[l] = new double[this.layers[l].getHiddenActivitiesNum()];
		}

		this.numWeights = numW;
		this.outputSize = is;
		this.zeroTargetValues = new double[is];
		this.rebufferOnDemand(1);

		this.myHiddenNeuronsSize = this.getHiddenActivitiesNum();
		// setting internal activity bounds....
		this.myHiddenNeuronsLowerBounds = new double[myHiddenNeuronsSize];
		this.myHiddenNeuronsUpperBounds = new double[myHiddenNeuronsSize];
		int idx = 0;
		for(int l=0; l < this.numLayers; l++) {
			double[] lb = new double[this.myHiddenActivities[l].length];
			double[] ub = new double[this.myHiddenActivities[l].length];
			this.layers[l].getInternalActivityBounds(lb, ub);
			Tools.copyToArray(lb, this.myHiddenNeuronsLowerBounds, idx);
			Tools.copyToArray(ub, this.myHiddenNeuronsUpperBounds, idx);
			idx += lb.length;
		}	
	}

	private static ANNLayer constructANNLayer(int inputSize, ANNLayer.LayerTypes layer, 
			int[] parLay, boolean[] biasesLay, ACT_FUNCT[] actFunctLay) {
		ANNLayer netTemp = null;
		switch(layer) {
		case LSTM: 
			netTemp = new ANNLayer_LSTM(inputSize, 
					parLay[0], parLay[1], parLay[2],//numHiddenCECCells, numCellsPerCEC, targets[0][0].length, 
					biasesLay, actFunctLay[0]);//new boolean[]{false, false, false},  ACT_FUNCT.Linear, wuType);
			break;
		default: System.err.println("Not supported: "+layer); return null;
		}
		return netTemp;
	}


	@Override
	public double[] getOutputActivities(int t) {
		return this.layers[this.numLayers-1].getOutputActivities(t);
	}

	@Override
	public double[] getInputBW(int t) {
		return this.layers[0].getInputBW(t);
	}

	@Override
	public void setOutputBW(double[] bwSignals, int timePoint) {
		this.layers[this.numLayers-1].setOutputBW(bwSignals, timePoint);
	}

	@Override
	public void rebufferOnDemand(int sequenceLength) {
		for (int l = 0; l < this.numLayers; l++) {
			this.layers[l].rebufferOnDemand(sequenceLength);
		}
		this.lastSequenceLength = 0;
		this.bufferlength = sequenceLength;
	}

	@Override
	public void resetAllActivitiesToZero() {
		for (int l = 0; l < this.numLayers; l++) {
			this.layers[l].resetAllActivitiesToZero();
		}
	}

	@Override
	public void resetFirstTimeStepActivitiesToZero() {
		for (int l = 0; l < this.numLayers; l++) {
			this.layers[l].resetFirstTimeStepActivitiesToZero();
		}
	}

	@Deprecated
	private void forwardPassZero(double[][] inputValues) {
		// this.resetFirstActivities(); // when working with forwardPassZero, this is not necessary!
		final int sequenceLength = inputValues.length;
		// starting at t=1 since in t=0 are the initialization values (zero values during training). 
		for (int t = 1; t <= sequenceLength; t++) {
			oneForwardPassAt(inputValues[t-1], t, t-1);
		}
		// Store input length of the current sequence.
		this.lastSequenceLength = sequenceLength;
	}	

	@Override
	public double[] forwardPass(double[] input) {
		assert(input.length == this.summedInputSize);
		// execute on self-recurrent step.
		this.oneForwardPassAt(input, 0, 0);
		// store & return output.
		this.lastSequenceLength = 1;
		return getOutputActivities(0);
	}

	@Override
	public void oneForwardPassAt(double[] input, int t, int prevt) {
		assert(input.length == this.summedInputSize);
		// moving iteratively forward through the layers.
		for (int l = 0; l < this.numLayers; l++) {
			this.layers[l].oneForwardPassAt(input, t, prevt);
			input = this.layers[l].getOutputActivities(t);
		}

	}

	private void backwardsPassZero(double[][] targetValues) {
		final int steps = this.lastSequenceLength;
		int t_target = targetValues.length - 1; // can be used to separate target length from sequence length... 
												// (e.g. if only the last n targets are defined).
		// compute reversely iteratively in time.
		double[] bwValues = new double[this.outputSize];
		for (int t = steps; t >= 0; t--,t_target--) {
			if (t_target >= 0) {
				double[] outputValues = this.layers[this.numLayers-1].getOutputActivities(t);
				for(int i=0; i<this.outputSize; i++) {
					bwValues[i] = outputValues[i] - targetValues[t_target][i]; 
				}
			}else{ // target length is used up... zero error coming from all other outputs
				bwValues = this.zeroTargetValues;
			}
			this.layers[this.numLayers-1].setOutputBW(bwValues, t);
			if(t == steps)
				this.oneBackwardPassAt(t, t-1, -1);
			else
				this.oneBackwardPassAt(t, t-1, t+1);
		}
		// ################################################
		// Compute the resulting weight derivatives.
		// ################################################
		setWeightsDerivativesGeneral(0, steps);
	}

	@Override
	public void oneBackwardPassAt(int t, int prevt, int nextt) {
		for (int l=this.numLayers-1; l >= 0; l--) {
			this.layers[l].oneBackwardPassAt(t, prevt, nextt);
			if(l>0)
				this.layers[l-1].setOutputBW(this.layers[l].getInputBW(t), t);
		}
	}


	@Override
	public void setWeightsDerivativesGeneral(int tInit, int tLast) {
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].setWeightsDerivativesGeneral(tInit, tLast);
		}		
	}

	@Override
	public void initializeWeights(Random rnd, double stddev) {
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].initializeWeights(rnd, stddev);
		}
	}

	@Override
	public int getNumWeights() {
		return this.numWeights;
	}

	public int[] getNumLayerWeights() {
		int[] ret = new int[this.numLayers];
		for(int l=0; l < this.numLayers; l++) {
			ret[l] = this.layers[l].getNumWeights();
		}
		return ret;
	}
	
	@Override
	public void readWeights(double[] myWeights) {
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].readWeights(this.myLayerWeights[l]);
		}
		Tools.map(this.myLayerWeights, myWeights);
	}

	public void readLayerWeights(double[][] myWeights) {
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].readWeights(myWeights[l]);
		}
	}

	@Override
	public void writeWeights(double[] myNewWeights) {
		Tools.map(myNewWeights, this.myLayerWeights);
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].writeWeights(this.myLayerWeights[l]);
		}		
	}

	@Override
	public void readDiffWeights(double[] myDiffWeights) {
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].readDiffWeights(this.myLayerDiffWeights[l]);
		}
		Tools.map(this.myLayerDiffWeights, myDiffWeights);		
	}

	@Override
	public int getHiddenActivitiesNum() {
		int numHAs = 0;
		for (int l=0; l < this.numLayers; l++) {
			numHAs += this.layers[l].getHiddenActivitiesNum();
		}
		return numHAs;
	}

	@Override
	public void getInternalActivityBounds(double[] lowerBounds, double[] upperBounds) {
		Tools.copyToArray(this.myHiddenNeuronsLowerBounds, lowerBounds);
		Tools.copyToArray(this.myHiddenNeuronsUpperBounds, upperBounds);
	}
	
	@Override
	public void readInternalActivities(double[] activities, int timePoint) {
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].readInternalActivities(this.myHiddenActivities[l], timePoint);
		}
		Tools.map(this.myHiddenActivities, activities);
	}
	
	@Override
	public void readBWErrors(double[] myCurrentBWErrors, int timePoint) {
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].readBWErrors(this.myHiddenBWErrors[l], timePoint);
		}
		Tools.map(this.myHiddenBWErrors, myCurrentBWErrors);
	}
	
	@Override
	public void writeInternalActivities(double[] activities, int timePoint) {
		Tools.map(activities, this.myHiddenActivities);
		for (int l=0; l < this.numLayers; l++) {
			this.layers[l].writeInternalActivities(this.myHiddenActivities[l], timePoint);
		}
	}

	@Override
	public int getOutputSize() {
		return this.outputSize;
	}
	
	@Override
	public int getInputSize() {
		return this.summedInputSize;
	}

	/**
	 * Stochastic gradient descent.
	 * 
	 * @param rnd Instance of Random.
	 * @param input Input vectors.
	 * @param target Target vectors.
	 * @param epochs Number of epochs.
	 * @param learningrate Value for the learning rate.
	 * @param momentumrate Value for the momentum rate.
	 * @param listener Listener to observe the training progress.
	 * @return The final epoch error.
	 */
	@Override
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
		final double[][] initialWeights = new double[this.numLayers][];
		// determining the maximum time series training sequence length. 
		int maxSequenceLength = 0;
		for(int i=0; i<input.length; i++) {
			int length = input[i].length; // 
			if(length > maxSequenceLength)
				maxSequenceLength = length;
		}
		// initializing weights and neural activity buffers.
		for(int l=0; l<this.numLayers; l++) {
			initialWeights[l] = new double[this.layers[l].getNumWeights()];
			this.layers[l].rebufferOnDemand(maxSequenceLength+1);
		}
		// initializing error based learners for each layer.
		ErrorBasedLearner[] ebl = new ErrorBasedLearner[this.numLayers];
		for(int l=0; l<this.numLayers; l++) {
			switch(weightUpdateType) {
			case GradientDescent:
				ebl[l] = new GradientDescentLearner(this.layers[l].getNumWeights(), learningRate, momentumRate);
				break;
			case Adam:
				ebl[l] = new AdamLearner(this.layers[l].getNumWeights(), learningRate);
				break;
            default:
                break;
			}
			layers[l].readWeights(initialWeights[l]);
			ebl[l].resetValues(initialWeights[l]);
		}
		// initializing index shuffling
		final int[] indices = new int[input.length];
		for (int i = 0; i < indices.length; i++) {
			indices[i] = i;
		}
		double error = 0;
		// epoch loop.
		for (int e = 0; e < epochs; e++) {
			// shuffle indices for this epoch.
			Tools.shuffle(indices, rnd);
			//
			double errorsum = 0.0;
			// train all samples in online manner, i.e. iterate over all samples
			// while considering the shuffled order and update the weights 
			// immediately after each sample
			for(int i=0; i<indices.length; i++) {
				if(e+1 == epochs && i+1 == indices.length) {
					System.out.println("Very last training instance.");
				}
				// do the actual training on the specific training instance indices[j]
				forwardPassZero(input[indices[i]]);
				// store & return full output.
				final double[][] output  = new double[this.lastSequenceLength][this.outputSize];
				final ANNLayer outLayer = this.layers[this.numLayers-1];
				for (int t = 0; t < this.lastSequenceLength; t++) {
					output[t] = outLayer.getOutputActivities(t+1).clone();
				}
				errorsum += Tools.MSE(output,  target[indices[i]]);

				backwardsPassZero(target[indices[i]]);

				for (int l=0; l < this.numLayers; l++) {
					this.layers[l].readDiffWeights(this.myLayerDiffWeights[l]);
					this.myLayerWeights[l] = ebl[l].learningIteration(this.myLayerDiffWeights[l]);
					this.layers[l].writeWeights(this.myLayerWeights[l]);
				}		
			}
			//
			error = errorsum / (double)(indices.length);
			if (listener != null) listener.afterEpoch(e + 1, error);
			if(e+1 == epochs) {
				System.out.println("Done with training");
			}
		}
		// done training the specified number of epochs... return the error achieved in the last epoch. 
		return error;
	}

	@Override
	@Deprecated
	public void forwardBackwardProject(double[] target, int depth, int numEpochs, double learningRate, double momentumRate,
			WEIGHT_UPDATE activeInferenceUpdateType) {
		//
		this.rebufferOnDemand(depth+1);
		// input is empty... I want to infer the hidden states. 
		double[][] zeroInput = new double[depth][this.summedInputSize];
		final int numHiddenActivities = this.getHiddenActivitiesNum();
		// 
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
//			double[] output = getOutputActivities(this.lastSequenceLength);
//			System.out.println("Current error (RMSE): "+Tools.RMSE(output, target));
			this.backwardsPassZero(new double[][]{target});
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

	
	
	/**
	 * Executes active inference within this network.
	 * 
	 * @param time
	 * @param actInfEBL
	 * @param actIIterations
	 * @param actIDepth
	 * @param actIMaintainTargetSteps
	 * @param outputToInputMap 
	 * @param currentContextInput -> context input vector (current top-down context estimate - assumed to be constant over time)
	 * @param sensorInputReadings -> sensor input over time (perceived in the past and imagined in the future)
	 * @param combinedInputReadings -> combined inputs over time (perceived in the past and imagined in the future)
	 * @param sensorOutputReadings -> sensory consequences over time (perceived in the past and imagined in the future)
	 * @param motorActivities -> motor activities over time (executed in the past and imagined in the future)
	 * @param bwMotorErrors -> motor errors from now until actIDepth (from previous optimization shifted).
	 * @param target
	 */
	public void doActiveMotorInference(final int time, 
			final ErrorBasedLearner actInfEBL, 
			final int actIIterations, final int actIDepth, final int actIMaintainTargetSteps,
			final OutputToInputMapping outputToInputMap, 
			final double[] currentContextInput,
			final double[][] sensorInputReadings, final double[][] combinedInputReadings, final double[][] sensorOutputReadings, 
			final double[][] motorActivities,
			final double[] target) {
		// #####################################################
		// begin of active-inference-based control process assumes that 
		//   1.) sensorInputReadings[time%bufferlength] gives the current sensory state perceived in the environment. 
		//   2.) net is in the state that it just -- at time "time" -- predicted 
		//         the next sensory state AFTER sensorInputReadings[time%bufferlength] and executing motorActivities[0].
		double[] bwInOutput = new double[sensorimotorSize];
		double[][] bwMotorErrors = new double[actIDepth][motorInputSize];
		double[] flatBwErrors = new double[actIDepth * motorInputSize];
		for(int iteration=0; iteration<actIIterations; iteration++) {
			// forward pass into the future...
			for(int t=1; t<actIDepth; t++) { // thinking through the imagined future motorActivity steps.
				sensorOutputReadings[(time+t-1)%bufferlength] = getOutputActivities((time+t-1)%bufferlength);
				outputToInputMap.mapStateAndOutputToNextInput(sensorInputReadings[(time+t-1)%bufferlength], 
						sensorOutputReadings[(time+t-1)%bufferlength], 
						sensorInputReadings[(time+t)%bufferlength]);
				Tools.map(sensorInputReadings[(time+t)%bufferlength], motorActivities[t], combinedInputReadings[(time+t)%bufferlength]);
				
				Tools.copyToArray(currentContextInput, 0, this.contextInputSize, combinedInputReadings[(time+t)%bufferlength], this.sensorimotorSize);
				
				// execute next prediction step given current sensor readings and motor command...
				oneForwardPassAt(combinedInputReadings[(time+t)%bufferlength], (time+t)%bufferlength, (time+t-1)%bufferlength);
			}
			sensorOutputReadings[(time+actIDepth-1)%bufferlength] = getOutputActivities((time+actIDepth-1)%bufferlength);
			outputToInputMap.mapStateAndOutputToNextInput(sensorInputReadings[(time+actIDepth-1)%bufferlength], 
					sensorOutputReadings[(time+actIDepth-1)%bufferlength], 
					sensorInputReadings[(time+actIDepth)%bufferlength]);
			//
			// back-prop-through time pass of active inference starts here!.... 
			for(int i=0; i<this.outputSize; i++) {
				bwInOutput[i] = sensorInputReadings[(time+actIDepth)%bufferlength][i] - target[i]; // goal of last step.
			}
			setOutputBW(bwInOutput, (time+actIDepth-1)%bufferlength);
			oneBackwardPassAt((time+actIDepth-1)%bufferlength, (time+actIDepth-2)%bufferlength, -1);
			// getting backwards error signal inputs.
			bwInOutput = getInputBW((time+actIDepth-1)%bufferlength);
			// extracting the motor-related error signals. 
			for(int j=sensorInputSize; j<sensorimotorSize; j++) // tapping onto the motor error!
				bwMotorErrors[actIDepth-1][j-sensorInputSize] = bwInOutput[j];
			// proceed with propagating the error backwards through time
			for(int t=actIDepth-2; t>=0; t--) {
				if(actIDepth-t <= actIMaintainTargetSteps) {
					for(int i=0; i<this.outputSize; i++) {
						bwInOutput[i] += (sensorInputReadings[(time+t+1)%bufferlength][i] - target[i]); // goal of last step.
					}
				}
				setOutputBW(bwInOutput, (time+t)%bufferlength);
				oneBackwardPassAt((time+t)%bufferlength, (time+t-1)%bufferlength, (time+t+1)%bufferlength);
				bwInOutput = getInputBW((time+t)%bufferlength);
				for(int j=sensorInputSize; j<sensorimotorSize; j++)
					bwMotorErrors[t][j-sensorInputSize] = bwInOutput[j];

			}
			// now adapt the imagined series of motor commands.
			Tools.map(bwMotorErrors, flatBwErrors);
			double[] flatMotorActivities = actInfEBL.learningIterationBounded(flatBwErrors, 0, 1);
			// Tools.bindArray(flatMotorActivities, 0, 1);
			Tools.map(flatMotorActivities, motorActivities);
			// predicting the one-step action consequences (to set the internal state of the network to its anticipatory state)
			//
			Tools.map(sensorInputReadings[time%bufferlength], motorActivities[0], combinedInputReadings[time%bufferlength]);
			Tools.copyToArray(currentContextInput, 0, this.contextInputSize, combinedInputReadings[time%bufferlength], this.sensorimotorSize);
			
			oneForwardPassAt(combinedInputReadings[time%bufferlength], time%bufferlength, (time-1)%bufferlength);
		}
	}	
	
	/**
	 * Executes context state and possibly hidden state inference. 
	 * Adapts the current context guess values.
	 * Regardless if hidden state inference is applied, 
	 * the hidden state at time time is affected also by context state inference!
	 * 
	 * @param time
	 * @param contextInfEBL
	 * @param contextIIterations
	 * @param contextIDepth
	 * @param hsInfEBL
	 * @param combinedInputReadings
	 * @param sensorOutputReadings
	 * @param currentContextGuess
	 */
	public void doHiddenStateAndContextInference(final int time,
			final ErrorBasedLearner contextInfEBL, 
			final int contextIIterations, final int contextIDepth,
			final ErrorBasedLearner hsInfEBL,
			final double[][] combinedInputReadings, final double[][] sensorOutputReadings, double[] currentContextGuess) {
		// ################################################
		// Starting Mode (& possibly also hidden state) Inference... 
			final double[] hsInfActErrors = new double[this.myHiddenNeuronsSize];
			for(int iteration=0; iteration<contextIIterations; iteration++) {
				double[] bwModeStateErrors = new double[this.contextInputSize];
				// backprop through time to infer mode
				for (int bpTime = time-1, count = contextIDepth-1; count >= 0; bpTime--, count--) {
					double[] outputValues = getOutputActivities(bpTime%this.bufferlength);
					for(int i=0; i<outputValues.length; i++) {
						outputValues[i] -= sensorOutputReadings[bpTime%this.bufferlength][i];
					}
					setOutputBW(outputValues, bpTime%this.bufferlength);
					if(bpTime == time-1) {
						oneBackwardPassAt(bpTime%this.bufferlength, (bpTime-1)%this.bufferlength, -1);
					}else{
						oneBackwardPassAt(bpTime%this.bufferlength, (bpTime-1)%this.bufferlength, (bpTime+1)%this.bufferlength);
					}
					Tools.addToArray(getInputBW(bpTime%this.bufferlength), sensorimotorSize, sensorimotorSize+this.contextInputSize, bwModeStateErrors);
				}
				// Adapt the current context input guess - i.e. context input inference.
				Tools.copyToArray(contextInfEBL.learningIterationBounded(bwModeStateErrors, 0, 1), currentContextGuess);
				// not necessary and actually disruptive: Tools.enforceProbabilityMass(currentModeGuess, 0, numTrainProblems);
				//
				if(hsInfEBL!=null) {
					readBWErrors(hsInfActErrors, (time-contextIDepth)%this.bufferlength);
					double[] hsInfActivities = hsInfEBL.learningIterationBounded(hsInfActErrors, 
							this.myHiddenNeuronsLowerBounds, this.myHiddenNeuronsUpperBounds);
					writeInternalActivities(hsInfActivities, (time-contextIDepth)%this.bufferlength);
				}
				//
				// adapt the previous forward predictions in the network accounting for the updated mode guess
				for(int t = contextIDepth-1; t>0; t--) {// the minus one is in order to enable hidden state adaptation
					Tools.copyToArray(currentContextGuess, 0, this.contextInputSize, combinedInputReadings[(time-t)%bufferlength], 
							this.sensorimotorSize);
					oneForwardPassAt(combinedInputReadings[(time-t)%this.bufferlength], (time-t)%this.bufferlength, (time-t-1)%this.bufferlength);
				}
			}

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

	public static ANN3InputComplexNet readNetwork(String fileName) throws IOException, ClassNotFoundException, ClassCastException {
		FileInputStream f = new FileInputStream(fileName);
		ObjectInputStream s = new ObjectInputStream(f);
		ANN3InputComplexNet ann = (ANN3InputComplexNet)s.readObject();
		s.close();
		f.close();
		return ann;
	}

}
