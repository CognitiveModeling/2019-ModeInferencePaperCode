package de.cogmod.anns.annevaluation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;
import java.util.Random;

import javax.swing.JFrame;

import de.cogmod.anns.annmodules.ANN3InputComplexNet;
import de.cogmod.anns.annmodules.ANNLayer.LayerTypes;
import de.cogmod.anns.errorlearners.AdamLearner;
import de.cogmod.anns.errorlearners.AdamLearnerSignDamping;
import de.cogmod.anns.errorlearners.ErrorBasedLearner;
import de.cogmod.anns.errorlearners.GradientDescentLearner;
import de.cogmod.problems.RB3Simulator;
import de.cogmod.problems.RB3Simulator.RB3Mode;
import de.cogmod.utilities.ActivationFunctions.ACT_FUNCT;
import de.cogmod.utilities.BasicLearningListener;
import de.cogmod.utilities.LearningListener;
import de.cogmod.utilities.RecordingLearningListener;
import de.cogmod.utilities.Tools;
import de.jannlab.math.Matrix;

/**
 * @author Sebastian Otte & Martin V. Butz
 */
public class RNNEval_CIEBdetection_MultiProblem {

	private final static long initialSeed = 42l;
	private final static Random rnd = new Random(initialSeed);
	
	// Hyper-Learning / Evaluation Coordination Parameters (multiple networks / multiple evaluation runs...)
	private static boolean doTrainMultipleNetworks = false;
	private static int numNetworksToTrain = 10;
	private static boolean doEvaluateMultipleNetworks = false;
	private static int numNetworksToEvaluate = 10;
	private static boolean doTrainAndEvaluateOneNetwork = true;
	
	private static boolean doTrainWITHOUTProvidingContext = true;
	private static boolean doRandomNextDynamicSystem = false;
	private static boolean doResetInitalContextGuess = false;
	
	// You may load the available network or build a new one, in that case set LoadNetwork to false
	// and set trainEpochsBlock to ex. 1000.
	private static boolean loadNetwork = true;
	private static String rnnFileName = "myRNN_LSTM16_clr.1_cif2_lr-4_lf30_sc205_doREpT_4";
	private static final int trainBuffer = 1001;
	
	// Number of training epochs: 
	private static int trainEpochsBlock = 0;
	private static final int stepsPerEpoch = 2001;
	private static final boolean doResetNetworkEachEpoch = true;
	// context switching
	private static final int switchContextFrequencyDefault =  205;
	private static final int switchContextFrequencyRange = 1;
	// potential annealing during extensive training (epoch blocks...)
	private static int annealSteps = 1;
	private static double annealFactor = 0.1;
	
	// MODEL INFERENCE ...
	private static final ErrorBasedLearner.WEIGHT_UPDATE wuType = ErrorBasedLearner.WEIGHT_UPDATE.Adam;
	private static final int weightUpdateFrequency = 30;
	private static final double initialLearningRate = 1.E-4;
	private static final double momentumrate = .95;
	//
	// ACTIVE CONTEXT STATE INFERENCE....
	private static ErrorBasedLearner.WEIGHT_UPDATE contextIUpdateTypeTrain = ErrorBasedLearner.WEIGHT_UPDATE.Adam;
	private static int contextIFreqDepthTrain = 10; // Frequency and depth of context inference.
	private static int contextIIterationsTrain = 5;
	private static double contextILearningRateTrain = 0.1;
	private static double contextIMomentumTrain = 0.9;
	private static double initialContextGuessValuesTrain = 1./3.;
	//
	// ACTIVE HIDDEN STATE INFERENCE....
	private static boolean doAdditionalHiddenStateInferenceTrain = false;
	private static ErrorBasedLearner.WEIGHT_UPDATE hsIUpdateTypeTrain = ErrorBasedLearner.WEIGHT_UPDATE.Adam;
	private static double hsILearningRateTrain = 0.01;
	private static double hsIMomentumTrain = 0.9;

	// Output error
	private static final int writeErrorFrequency = 10;

	private static final RB3Mode[] trainProblems = {RB3Mode.Rocket, RB3Mode.Stepper, RB3Mode.Glider};

	private static int currentProblemIndex;
	private static RB3Mode currentTrainProblem;
	private static final int numTrainProblems = trainProblems.length;
	private static final double[] currentProblemArray = new double[numTrainProblems];
	private static final int switchProblemEpochFrequency = 1;

	private static final boolean doPredictDeltas = true;

	private static final boolean doVisTrain = false;
	private static final int sleepVisTrain = 0;

	private static final LayerTypes[] usedLayers = {LayerTypes.LSTM};
	
	// parameter array: 
	private static final int[][] parametersForEachLayer = {{16, 1, 2}};
	// bias:
	private static final boolean[][] biasesInLayers = {{false,false,false}};
    // Activation function array:
	private static final ACT_FUNCT[][] actFuncts = {{ACT_FUNCT.Linear}};

	// SETTINGS FOR CONTROLLING ACTIVE (GOAL-DIRECTED) MOTOR INFERENCE:
	
	// ACTIVE MOTOR INFERENCE....
	private static ErrorBasedLearner.WEIGHT_UPDATE actIUpdateType = ErrorBasedLearner.WEIGHT_UPDATE.Adam;
	private static int actIDepth = 7;
	private static int actIMaintainTargetSteps = 7;
	private static int actIIterations = 20;
	private static double actILearningRate = 0.1;
	private static double actIMomentum = 0.9;
	//
	// ACTIVE CONTEXT MODE STATE INFERENCE....
	private static boolean doContextStateInferenceControl = true; // NOTE: false implies that the mode indicator is set correctly ("external" information)
	private static ErrorBasedLearner.WEIGHT_UPDATE contextIUpdateTypeControl = ErrorBasedLearner.WEIGHT_UPDATE.Adam;
	private static int contextIDepthControl = 15;
	private static int contextIIterationsControl = 15;
	private static double contextILearningRateControl = 0.1;
	private static double contextIMomentumControl = 0.9;
	private static double initialContextGuessValuesControl = 1./3.;
	//
	// ACTIVE HIDDEN STATE INFERENCE....
	private static boolean doAdditionalHiddenStateInferenceControl = false;
	private static ErrorBasedLearner.WEIGHT_UPDATE hsIUpdateTypeControl = ErrorBasedLearner.WEIGHT_UPDATE.Adam;
	private static double hsILearningRateControl = 0.01;
	private static double hsIMomentumControl = 0.9;
	//
	// setup of experimental parameters... 
	private static final boolean doVisActInf = true;
	private static final int sleepVisActInf = 0;
	// exact evaluation numbers and mode switching frequency
	private static final int actISwitchTargetFrequency = 150;
	private static final int actIEvaluateIteractions = 15001;
	//    
	@SuppressWarnings("unused")
	public static void main(String[] args) throws IOException {
		currentProblemIndex = 0;
		currentProblemArray[currentProblemIndex] = 1;
		currentTrainProblem = trainProblems[currentProblemIndex];
		RB3Simulator simulator = createSimulator();
		if(doVisTrain || doVisActInf)
			activateVisualization(simulator);
		//
		if(doTrainMultipleNetworks) {
			trainMultipleNetworks(simulator);
		}
		if(doEvaluateMultipleNetworks) {
			evaluateMultipleNetworks(simulator);
		}
		if(doTrainAndEvaluateOneNetwork) {
			// 
			// Network setup...
			//
			final ANN3InputComplexNet net = (ANN3InputComplexNet)loadOrCreateANN(simulator);

			if(trainEpochsBlock > 0) {
				if(doTrainWITHOUTProvidingContext) {
					System.out.println("Starting to train the network...");		
					trainNeuralNetworkWithoutContextInput(net, simulator, initialLearningRate, 
							new BasicLearningListener(writeErrorFrequency));
					// saving the file if sufficient learning was done
					if(trainEpochsBlock*stepsPerEpoch*annealSteps >= 100000) // only save the learning effect when sufficient training was done.
						net.writeNetwork(rnnFileName);
				}else{
					System.err.println("Training with Context Information / witout context state inference is not supported anylonger...");		
					return;
				}
			}
			// #############################################################################
			System.out.println("Starting active-inference-based control... ");		
			activeControlandMotorInference(net, simulator, 1, new BasicLearningListener(1));
		}
	}

	private static void trainNeuralNetworkWithoutContextInput(ANN3InputComplexNet net,
			RB3Simulator simulator,
			double learningRate, 
			LearningListener listener) {
		//
		// Training setup...
		//
		net.rebufferOnDemand(trainBuffer);
		//		
		int numWeights = net.getNumWeights();
		final double[] initialWeights = new double[numWeights];
		final double[] dweights      = new double[numWeights];
		//
		// initializing error based learner
		//
		ErrorBasedLearner ebl = null;
		switch(wuType) {
		case GradientDescent:
			ebl = new GradientDescentLearner(numWeights, learningRate, momentumrate);
			break;
		case Adam:
			ebl = new AdamLearner(numWeights, learningRate);
			break;
		case AdamLearnerSignDamping:
			ebl = new AdamLearnerSignDamping(numWeights, learningRate);
			break;
		}
		net.readWeights(initialWeights);
		ebl.resetValues(initialWeights);
		//
		// context state inference setup
		final int contextSize = numTrainProblems;
		// initiate the context inference adaptation mechanism (error-based learner)
		ErrorBasedLearner contextInfEBLTrain = null;
		switch(contextIUpdateTypeTrain) {
		case GradientDescent:
			contextInfEBLTrain = new GradientDescentLearner(contextSize, contextILearningRateTrain, contextIMomentumTrain);
			break;
		case Adam:
			contextInfEBLTrain = new AdamLearner(contextSize, contextILearningRateTrain);//, .999 , .9999);
			break;
		case AdamLearnerSignDamping:
			contextInfEBLTrain = new AdamLearnerSignDamping(contextSize, contextILearningRateTrain);//, .999 , .9999);
			break;
		}
		// current context guess is initially set to uniform values
		double[] currentContextGuess = new double[contextSize];
		Tools.setValArray(currentContextGuess, 0, contextSize, initialContextGuessValuesTrain);
		contextInfEBLTrain.resetValues(currentContextGuess);
		//
		// hidden state inference
		final int hiddenStateSize = net.getHiddenActivitiesNum();
		double[] minBoundHiddenActivities = new double[hiddenStateSize];
		double[] maxBoundHiddenActivities = new double[hiddenStateSize];
		net.getInternalActivityBounds(minBoundHiddenActivities, maxBoundHiddenActivities);
		ErrorBasedLearner hsInfEBLTrain = null;
		if(doAdditionalHiddenStateInferenceTrain) {
			switch(hsIUpdateTypeTrain) {
			case GradientDescent:
				hsInfEBLTrain = new GradientDescentLearner(hiddenStateSize, hsILearningRateTrain, hsIMomentumTrain);
				break;
			case Adam:
				hsInfEBLTrain = new AdamLearner(hiddenStateSize, hsILearningRateTrain);//, .999 , .9999);
				break;
			case AdamLearnerSignDamping:
				hsInfEBLTrain = new AdamLearnerSignDamping(hiddenStateSize, hsILearningRateTrain);//, .999 , .9999);
				break;
			}
			// hidden states are set to zero initially.
			hsInfEBLTrain.resetValues(new double[hiddenStateSize]);
		}
		//
		//
		final int sensorInputSize = simulator.getSensorInputSize();
		final int motorSize = simulator.getMotorSize();
		final int sensorimotorSize = sensorInputSize + motorSize;
		final int combinedInputSize = sensorimotorSize + contextSize;
		final int sensorOutputSize = simulator.getSensorOutputSize();
		//
		double[][] sensorInputReadings = new double[trainBuffer][sensorInputSize];
		double[][] sensorOutputReadings = new double[trainBuffer][sensorOutputSize];
		double[][] combinedInputReadings = new double[trainBuffer][combinedInputSize];
		//
		// start the training.
		//
		// start acting and learning
		simulator.setDoDraw(doVisTrain);
		//
		//
		//
		for(int e=0; e<trainEpochsBlock; e++) {
			simulator.reset();
			if(doResetNetworkEachEpoch)
				net.resetAllActivitiesToZero();
			double[] motorCommands = new double[simulator.getMotorSize()];
			// zero motor command and first sensor reading for initialization purposes... 
			simulator.executeAndGet(motorCommands, sensorOutputReadings[0], sensorInputReadings[0]);
			if(doVisTrain) {
				try {
					Thread.sleep(sleepVisTrain);
				} catch (InterruptedException exception) {
					exception.printStackTrace();
				}	
			}
			Tools.map(sensorInputReadings[0], motorCommands, currentContextGuess, combinedInputReadings[0]);
			// setting network to predict next input.
			net.oneForwardPassAt(combinedInputReadings[0], 0, trainBuffer-1);			
			double errorsum = 0;
			double errorsumInferredEventHidden = 0;
			double errorsumInferredOnlineHidden = 0;
			//
			// entering / starting each loop iteration with predictive state of mind and ready-to-execute motor command
			int nextSwitchTime = switchContextFrequencyDefault + 
					rnd.nextInt(switchContextFrequencyRange * 2) - switchContextFrequencyRange;
			for(int time=1, tempTime=1; time<=stepsPerEpoch; time++,tempTime++) {
				// execute the determined motor command (note: second step still zero motor command for initialization purposes) 
				simulator.executeAndGet(motorCommands, sensorOutputReadings[(time-1)%trainBuffer], 
						sensorInputReadings[time%trainBuffer]);
				// compare prediction with actual outcome... .
				double err = Tools.RMSE(net.getOutputActivities((time-1)%trainBuffer), sensorOutputReadings[(time-1)%trainBuffer]);
				errorsum += err;
				// check for backwards pass... 
				if(tempTime >= nextSwitchTime) {
					tempTime = 0;
					nextSwitchTime = switchContextFrequencyDefault + rnd.nextInt(switchContextFrequencyRange*2) - switchContextFrequencyRange;
				// finally.. switch the context... 
					setAndActivateNextDynamicSystem(simulator);
					if(doResetInitalContextGuess) {
						Tools.setValArray(currentContextGuess, 0, contextSize, initialContextGuessValuesTrain);
						contextInfEBLTrain.resetValues(currentContextGuess);
					}
				} 
				// #########################################################
				// next: possibly conduct local context adaptation.
				// #########################################################
				if(time % contextIFreqDepthTrain == 0) { // active model inference within an event... 
					net.doHiddenStateAndContextInference(time, contextInfEBLTrain, 
							contextIIterationsTrain, contextIFreqDepthTrain,
							hsInfEBLTrain, 
							combinedInputReadings, sensorOutputReadings, currentContextGuess);
					// compute resulting error estimates... 
					for(int bptime = time-contextIFreqDepthTrain; bptime<=time; bptime++) {
						double errTemp = Tools.RMSE(net.getOutputActivities(bptime%trainBuffer), 
								sensorOutputReadings[bptime%trainBuffer]);
						errorsumInferredOnlineHidden += errTemp;
					}
				}
				if(time % weightUpdateFrequency == 0) {
					// backprop through time... model weight inference...
					for (int bpTime = time-1, count=weightUpdateFrequency; count > 0; bpTime--,count--) {
						double[] outputValues = net.getOutputActivities(bpTime%trainBuffer);
						for(int i=0; i<outputValues.length; i++) {
							outputValues[i] -= sensorOutputReadings[bpTime%trainBuffer][i];
						}
						net.setOutputBW(outputValues, bpTime%trainBuffer);
						if(bpTime == time-1) {
							net.oneBackwardPassAt(bpTime%trainBuffer, (bpTime-1)%trainBuffer, -1);
						}else{
							net.oneBackwardPassAt(bpTime%trainBuffer, (bpTime-1)%trainBuffer, (bpTime+1)%trainBuffer);
						}
					}
					// ################################################
					// Compute the resulting weight derivatives.
					// ################################################
					net.setWeightsDerivativesGeneral((time-weightUpdateFrequency+1)%trainBuffer, (time-1)%trainBuffer);
					net.readDiffWeights(dweights);
					final double[] newWeights = ebl.learningIteration(dweights);
					net.writeWeights(newWeights);		
				}
				// motor command determination...
				setMotorCommand(motorCommands);
				Tools.map(sensorInputReadings[time%trainBuffer], motorCommands, currentContextGuess, combinedInputReadings[time%trainBuffer]);
				// execute next prediction step given current sensor readings and motor command...
				net.oneForwardPassAt(combinedInputReadings[time%trainBuffer], time%trainBuffer, (time-1)%trainBuffer);
			}
			//
			double error = errorsum / (double)(stepsPerEpoch);
			double errorInferred = errorsumInferredEventHidden / (double)stepsPerEpoch;
			double errorInferredOnline = errorsumInferredOnlineHidden / (double)stepsPerEpoch;
			if (listener != null) 
				listener.afterEpoch(e + 1, error, errorInferred, errorInferredOnline);
			if(e+1 == trainEpochsBlock) {
				System.out.println("Done with training");
			}
			if( (e+1) % switchProblemEpochFrequency == 0) {
				setAndActivateNextDynamicSystem(simulator);
			}
		} // end of one epoch
	}


	/**
	 * Active control mode and motor inference - evaluation. 
	 * 
	 * @param net network with proper current state in time. 
	 * @param simulator the control problem in the according state
	 * @param time current time for which motor activity is to-be inferred.
	 * @throws IOException 
	 */
	@SuppressWarnings("unused")
	private static void activeControlandMotorInference(ANN3InputComplexNet net, RB3Simulator simulator, int time, LearningListener listener) throws IOException {
		assert(actIDepth+contextIDepthControl < trainBuffer);
		// resetting everything to get started from "base"
		//
		simulator.setDoDraw(doVisActInf);
		simulator.reset();
		//
		net.rebufferOnDemand(trainBuffer);
		net.resetAllActivitiesToZero();		
		//
		final int sensorInputSize = simulator.getSensorInputSize();
		final int motorSize = simulator.getMotorSize();
		final int sensorimotorSize = sensorInputSize + motorSize;
		final int sensorOutputSize = simulator.getSensorOutputSize();
		//
		// motor inference setup
		final int motorAdaptationSize = actIDepth * motorSize;
		ErrorBasedLearner actInfEBL = null;
		switch(actIUpdateType) {
		case GradientDescent:
			actInfEBL = new GradientDescentLearner(motorAdaptationSize, actILearningRate, actIMomentum);
			break;
		case Adam:
			actInfEBL = new AdamLearner(motorAdaptationSize, actILearningRate);
			break;
		case AdamLearnerSignDamping:
			actInfEBL = new AdamLearnerSignDamping(motorAdaptationSize, actILearningRate);
			break;
		}
		double[][] motorActivities = new double[actIDepth][motorSize];
		actInfEBL.resetValues(new double[motorAdaptationSize]);
		//
		// hidden state inference
		final int hsInferenceSize = net.getHiddenActivitiesNum();
		double[] minBoundHiddenActivities = new double[hsInferenceSize];
		double[] maxBoundHiddenActivities = new double[hsInferenceSize];
		net.getInternalActivityBounds(minBoundHiddenActivities, maxBoundHiddenActivities);
		ErrorBasedLearner hsInfEBL = null;
		if(doAdditionalHiddenStateInferenceControl) {
			switch(hsIUpdateTypeControl) {
			case GradientDescent:
				hsInfEBL = new GradientDescentLearner(hsInferenceSize, hsILearningRateControl, hsIMomentumControl);
				break;
			case Adam:
				hsInfEBL = new AdamLearner(hsInferenceSize, hsILearningRateControl);//, .999 , .9999);
				break;
			case AdamLearnerSignDamping:
				hsInfEBL = new AdamLearnerSignDamping(hsInferenceSize, hsILearningRateControl);//, .999 , .9999);
				break;
			}
			hsInfEBL.resetValues(new double[hsInferenceSize]);
		}
		//
		// context state inference setup
		final int contextInferenceSize = numTrainProblems;
		// initiate the mode inference adaptation mechanism (error-based learner)
		ErrorBasedLearner contextInfEBL = null;
		switch(contextIUpdateTypeControl) {
		case GradientDescent:
			contextInfEBL = new GradientDescentLearner(contextInferenceSize, contextILearningRateControl, contextIMomentumControl);
			break;
		case Adam:
			contextInfEBL = new AdamLearner(contextInferenceSize, contextILearningRateControl);//, .999 , .9999);
			break;
		case AdamLearnerSignDamping:
			contextInfEBL = new AdamLearnerSignDamping(contextInferenceSize, contextILearningRateControl);//, .999 , .9999);
			break;
		}
		// The current context guess, still empty
		double[] currentContextGuess = new double[contextInferenceSize];
		Tools.setValArray(currentContextGuess, 0, numTrainProblems, initialContextGuessValuesControl);
		contextInfEBL.resetValues(currentContextGuess);
		if(!doContextStateInferenceControl) {
			Tools.copyToArray(currentProblemArray, currentContextGuess); // no context inference -> set it to the correct value!
		}
		//
		double[][] sensorInputReadings = new double[trainBuffer][sensorInputSize];
		double[][] sensorOutputReadings = new double[trainBuffer][sensorOutputSize];
		double[][] combinedInputReadings = new double[trainBuffer][sensorimotorSize + numTrainProblems];
		// entrain network to base location.
		// zero motor command and first sensor reading for initialization purposes...
		simulator.executeAndGet(motorActivities[0], sensorOutputReadings[time%trainBuffer], sensorInputReadings[time%trainBuffer]);
		//
		Tools.map(sensorInputReadings[time%trainBuffer], motorActivities[0], currentContextGuess, combinedInputReadings[time%trainBuffer]);
		//
		// setting network to predict next input.
		net.oneForwardPassAt(combinedInputReadings[time%trainBuffer], time%trainBuffer, (time-1)%trainBuffer);
		sensorOutputReadings[time%trainBuffer] = net.getOutputActivities(time%trainBuffer);
		//
		time++;
		// now set the environment to the next state... (executing another zero action) for entrainment...
		simulator.executeAndGet(motorActivities[0], sensorOutputReadings[(time-1)%trainBuffer], sensorInputReadings[time%trainBuffer]);// sensorOutputReading ignored here.
		// setting network to predict next input.
		Tools.map(sensorInputReadings[time%trainBuffer], motorActivities[0], currentContextGuess, combinedInputReadings[time%trainBuffer]);
		//
		net.oneForwardPassAt(combinedInputReadings[time%trainBuffer], time%trainBuffer, (time-1)%trainBuffer);			
		//
		// target initialization 
		//
		double[] target = new double[]{0,1};
		simulator.setCurrentTarget(target);
		System.out.println("Current Target: " + Tools.toStringArrayPrecision(target, 2));
		//
		int startTime = time-1;
		final double targetError = 0.005;
		double targetErrorReached = 0;
		double targetErrorReachedTotal = 0;
		//
		double errorAccumulation = 0;
		double errorAccTotal = 0;
		//
		double finalPrecision = 0;
		//
		int numTargets = 1;
		
		// #####################################################################
		// #### START OF: 
		//      Active inference and context inference evaluation
		while(time-startTime < actIEvaluateIteractions) {
			if(doVisActInf) {
				try {
					Thread.sleep(sleepVisActInf);
				} catch (InterruptedException exception) {
					exception.printStackTrace();
				}
			}
			if(time % actISwitchTargetFrequency == 0) { // set a new target 
				double[] output = net.getOutputActivities((time+actIDepth-1)%trainBuffer);
				double currentErr = Tools.RMSE(sensorInputReadings[time%trainBuffer], target);
				finalPrecision += currentErr;
				errorAccTotal += errorAccumulation / actISwitchTargetFrequency;
				targetErrorReachedTotal += targetErrorReached / actISwitchTargetFrequency;

				target = new double[]{rnd.nextDouble()*1.5-.75,.25+1.5*rnd.nextDouble()};					
				simulator.setCurrentTarget(target);
				System.out.println("Current Target (#"+numTargets+"): " + Tools.toStringArrayPrecision(target, 2)+ 
						" --- Previous Target final error: "+currentErr);

				switch(currentContextGuess.length) {
				case 1: default:
					listener.afterEpoch(numTargets, currentErr, 
							targetErrorReached / actISwitchTargetFrequency, errorAccumulation / actISwitchTargetFrequency);
					break;
				case 2:
					listener.afterEpoch(numTargets,	currentErr, 
							targetErrorReached / actISwitchTargetFrequency, errorAccumulation / actISwitchTargetFrequency,
							(double)currentProblemIndex,currentContextGuess[0],currentContextGuess[1]);
					break;
				case 3:
					listener.afterEpoch(numTargets, currentErr, 
							targetErrorReached / actISwitchTargetFrequency, errorAccumulation / actISwitchTargetFrequency,
							(double)currentProblemIndex,currentContextGuess[0],currentContextGuess[1],currentContextGuess[2]);
					break;
				case 4:
					listener.afterEpoch(numTargets, currentErr, 
							targetErrorReached / actISwitchTargetFrequency, errorAccumulation / actISwitchTargetFrequency,
							(double)currentProblemIndex,currentContextGuess[0],currentContextGuess[1],currentContextGuess[2],currentContextGuess[3]);
					break;
				}
				errorAccumulation = 0;
				targetErrorReached = 0;
				numTargets++;
				if(numTargets % switchProblemEpochFrequency == 0) { // switching control problem
					setAndActivateNextDynamicSystem(simulator);
				}
			}
			//
			// ACTIVE MOTOR INFERENCE starts here ############################
			//
			net.doActiveMotorInference(time, actInfEBL, actIIterations, actIDepth, actIMaintainTargetSteps,
					simulator, 
					currentContextGuess,
					sensorInputReadings, combinedInputReadings, sensorOutputReadings, 
					motorActivities, 
					target);
			//
			// ... NEXT time step... 
			time++;
			//
			// Visualization... 
			if(doVisActInf) {
				Matrix m = new Matrix(actIDepth,2);
				double x = sensorInputReadings[time%trainBuffer][0];
				double y = sensorInputReadings[time%trainBuffer][1];
				for(int t=0; t<actIDepth; t++) {
					double[] out = net.getOutputActivities((time+t)%trainBuffer);
					x += out[0];
					y += out[1];
					m.set(t, 0, x);
					m.set(t, 1, y);
				}
				simulator.setStateSeqshort(m);
			}
			// execute the ACTUAL behavior getting the next sensory input state, which is thus the current one (therefore the 0 index!!!). 
			simulator.executeAndGet(motorActivities[0], sensorOutputReadings[(time-1)%trainBuffer], sensorInputReadings[time%trainBuffer]);
			//
			// error monitoring...
			double currentError = Tools.RMSE(sensorInputReadings[time%trainBuffer], target);
			errorAccumulation += currentError;
			if(currentError <= targetError)
				targetErrorReached++;
			//
			// ################################################
			// Starting context (& possibly also hidden state) Inference... 
			if( (doContextStateInferenceControl || doAdditionalHiddenStateInferenceControl) && (time-startTime) > contextIDepthControl) {
				net.doHiddenStateAndContextInference(time, contextInfEBL, 
						contextIIterationsControl, contextIDepthControl,
						hsInfEBL, 
						combinedInputReadings, sensorOutputReadings, currentContextGuess);
			}
			// shift the motor activities... 
			for(int i=0; i<actIDepth-1; i++) {
				for(int j=0; j<motorSize; j++) {
					motorActivities[i][j] = motorActivities[i+1][j];
				}
			}
			actInfEBL.shiftValues(motorSize);
			// predict results of next (imagined) motor command (note: values in motorActivities array was shifted!)
			Tools.map(sensorInputReadings[time%trainBuffer], motorActivities[0], currentContextGuess, combinedInputReadings[time%trainBuffer]);
			if(time % actISwitchTargetFrequency == 0) { // reporting the mode guess before switching to the next target (and probably the control mode) 
				System.out.println("Mode guess: "+Tools.toStringArrayPrecision(currentContextGuess, 2)+" with current control="+currentProblemIndex);
			}
			//
			net.oneForwardPassAt(combinedInputReadings[time%trainBuffer], time%trainBuffer, (time-1)%trainBuffer);
		}
		
		// active inference evaluation loop
		//
		// reporting error measurements:
		System.out.println("ErrAccum: "+errorAccTotal / numTargets
				+ " TargetErrorReached: " 
				+ targetErrorReachedTotal / numTargets 
				+ " FinalPrecision: "
				+ finalPrecision / numTargets);
	} // done with active inference evaluation routine.

	private static void setAndActivateNextDynamicSystem(RB3Simulator simulator) {
		if(doRandomNextDynamicSystem) {
			currentProblemIndex += 1 + rnd.nextInt(numTrainProblems-1);
		}else{
			currentProblemIndex ++;
		}
		currentProblemIndex %= numTrainProblems;
		currentTrainProblem = trainProblems[currentProblemIndex];
		simulator.setCurrentMode(currentTrainProblem);
		for(int i=0; i<numTrainProblems; i++)
			currentProblemArray[i] = 0;
		currentProblemArray[currentProblemIndex] = 1;
	}

	private static void evaluateMultipleNetworks(RB3Simulator simulator) throws IOException {
		//		
		String baseFileName = new String(rnnFileName);
		doRandomNextDynamicSystem = false; // using a deterministic sequence of vehicles here to assure comparability.
		double[][][] errorValues = new double[numNetworksToEvaluate][][];
		int numActInfRuns = 0;
		for(int i=0; i<numNetworksToEvaluate; i++) {
			simulator.reset();
			loadNetwork = true;
			rnnFileName = baseFileName+i;
			System.out.println("################-------------  BEGINNING to EVALUATE network from file: "+rnnFileName+" ------------############################################");
			ANN3InputComplexNet net = (ANN3InputComplexNet)loadOrCreateANN(simulator);		
			RecordingLearningListener listener = new RecordingLearningListener(7,1);
			boolean done = false;
			doContextStateInferenceControl = true;
			doAdditionalHiddenStateInferenceControl = false;
			contextILearningRateControl = 0.1;
			hsILearningRateControl = 0.1;
			while(!done) {
				System.out.println("----------"+rnnFileName+"------------------- >>>>>>>>>>>  Evaluating Setting: "+
						doContextStateInferenceControl+ " " +contextILearningRateControl+
						" "+doAdditionalHiddenStateInferenceControl+" "+hsILearningRateControl); 
				numActInfRuns++;
				rnd.setSeed(initialSeed); // ensuring that each network is tested on the identical sequence of target locations.
				currentProblemIndex = 0;
				for(int ii=0; ii<currentProblemArray.length; ii++)
					currentProblemArray[ii] = 0;
				currentProblemArray[currentProblemIndex] = 1;
				currentTrainProblem = trainProblems[currentProblemIndex];
				simulator.setCurrentMode(currentTrainProblem);
				//
				activeControlandMotorInference(net, simulator, 1, listener);
				if(!doContextStateInferenceControl && !doAdditionalHiddenStateInferenceControl) {
					doContextStateInferenceControl = true;
				}else{
					if(doContextStateInferenceControl) {
						if(contextILearningRateControl > .000101 && 
								(!doAdditionalHiddenStateInferenceControl || 
										contextILearningRateControl > .101 * hsILearningRateControl)) {
							contextILearningRateControl *= 0.1;
						}else{
							if(!doAdditionalHiddenStateInferenceControl) {
								doAdditionalHiddenStateInferenceControl = true;
								contextILearningRateControl = 0.1;
								hsILearningRateControl = 0.1;
							}else{
								contextILearningRateControl = hsILearningRateControl;
								if(hsILearningRateControl > .000101) {
									hsILearningRateControl *= .1;
								}else{
									done = true;
								}
							}
						}
					}
				}
			}
			errorValues[i] = listener.getRecordedErrors();
		}
		numActInfRuns /= numNetworksToEvaluate;
		System.out.println("Evaluated "+numActInfRuns+" different settings per network!");
		writePerformanceDataToFile(baseFileName+"PerfVals_miD_"+contextIDepthControl+"_miI_"+contextIIterationsControl+".tab", errorValues);
		rnnFileName = baseFileName;
	}

	private static void trainMultipleNetworks(RB3Simulator simulator) {
		//		
		String baseFileName = new String(rnnFileName);

		double[][][] errorValues = new double[numNetworksToTrain][][];
		for(int i=0; i<numNetworksToTrain; i++) {
			simulator.reset();
			double currentLR = initialLearningRate;
			loadNetwork = false;
			rnnFileName = baseFileName+i;
			System.out.println("Beginning training of network for file: "+rnnFileName);
			ANN3InputComplexNet net = loadOrCreateANN(simulator);		
			RecordingLearningListener listener = new RecordingLearningListener(3,1);
			for(int j=0; j<RNNEval_CIEBdetection_MultiProblem.annealSteps; j++) {
				if(doTrainWITHOUTProvidingContext) {
					trainNeuralNetworkWithoutContextInput(net, simulator, currentLR, listener);
				}else{
					System.err.println("Training with Context Information is not supported anylonger...");		
					return;
				}
				currentLR *= RNNEval_CIEBdetection_MultiProblem.annealFactor;
			}
			errorValues[i] = listener.getRecordedErrors();
			// saving the file if sufficient learning was done
			if(trainEpochsBlock*stepsPerEpoch*annealSteps >= 10000) // only save the learning effect when sufficient training was done.
				net.writeNetwork(rnnFileName);
		}
		writePerformanceDataToFile(baseFileName+"ErrorValues.tab", errorValues);
		rnnFileName = baseFileName;
	}

	private static void writePerformanceDataToFile(String fileName, double[][][] performanceValues) {
		// Writing the recorded Error Values during learning to the corresponding file...
		File outFile = new File(fileName);
		FileWriter fw;
		try {
			fw = new FileWriter(outFile);
			BufferedWriter bw =new BufferedWriter(fw);
			for(int j=0; j<performanceValues[0].length; j++) {
				bw.write(""+(j+1));
				for(int i=0; i<performanceValues.length; i++) {
					bw.write(" ");
					for(int k=0; k<performanceValues[i][j].length; k++) {
						bw.write(" "+String.format(Locale.US, "%.10f", performanceValues[i][j][k]));
					}
				}
				bw.newLine();
			}
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void activateVisualization(RB3Simulator simulator) {
		final String caption   = simulator.getProblemName();
		final JFrame frame     = new JFrame(caption);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.add(simulator.getPanel());
		frame.setResizable(false);
		frame.pack();
		simulator.activateKeyListener(frame);
		frame.setVisible(true);
	}

	private static RB3Simulator createSimulator() {
		//
		// Problem setup... 
		//
		final double fps     = 30.0;
		final double dtmsec  = 1000.0 / fps;
		final double dtsec   = dtmsec / 1000.0;
		RB3Simulator simulator = 
				new RB3Simulator(currentTrainProblem, doPredictDeltas, 
						0.1, 0.06, 1.2, -1.5, 1.5, 2.0, dtsec, new double[]{0,0});
		return simulator;
	}

	private static ANN3InputComplexNet loadOrCreateANN(RB3Simulator simulator) {
		ANN3InputComplexNet netTemp = null;
		if(loadNetwork) {
			try{
				netTemp = ANN3InputComplexNet.readNetwork(rnnFileName);
				System.out.println("Loaded complex network with "+netTemp.getNumWeights()+" weights.");
			}catch(IOException | ClassNotFoundException | ClassCastException e) {
				System.err.println("Failed to read Network with name "+rnnFileName+"! ..."+e);
			}
		}
		if(netTemp==null) {
			System.out.println("Generating a new complex ANN network!");
			netTemp = new ANN3InputComplexNet(simulator.getSensorInputSize(), simulator.getMotorSize(), numTrainProblems,  
					usedLayers, parametersForEachLayer, biasesInLayers, actFuncts);				
			// generate initial weights 
			netTemp.initializeWeights(rnd, 0.1);
			System.out.println("Created complex network with "+netTemp.getNumWeights()+" weights.");
		}
		return netTemp;
	}

	/**
	 * random motor command sampling - -slightly problem dependent.
	 * 
	 * @param motorCommands the array within which the motor commands are set. 
	 */
	private static void setMotorCommand(double[] motorCommands) {
		switch(currentTrainProblem) {
		case Rocket: case Stepper: case Glider: case GliderGravity: case Car:
			if(rnd.nextDouble() < .7) { // switching prob
				if(rnd.nextDouble() < 0.05) { // 5% exception: do nothing 
					for(int i=0; i<motorCommands.length; i++) {
						motorCommands[i] = 0;
					}
				}else{
					if(rnd.nextDouble() < 0.2 && currentTrainProblem!=RB3Mode.Car) { // 20% do the same
						motorCommands[0] = 1-rnd.nextDouble()*rnd.nextDouble();						
						motorCommands[1] = motorCommands[0];
					}else{
						for(int i=0; i<motorCommands.length; i++) {
							// preferring higher activities while sampling
							motorCommands[i] = rnd.nextDouble();
						}
					}
				}
			}
			break;
		case Stepper2Dir:
			if(rnd.nextDouble() < .7) { // switching prob
				if(rnd.nextDouble() < 0.05) { // 5% exception: do nothing 
					for(int i=0; i<motorCommands.length; i++) {
						motorCommands[i] = 0;
					}
				}else{
					if(rnd.nextDouble() < 0.2) { // 20% same thrust to both
						motorCommands[0] = 1-rnd.nextDouble()*rnd.nextDouble();						
						motorCommands[1] = motorCommands[0];
						motorCommands[2] = motorCommands[3] = 0;
					}else{
						for(int i=0; i<motorCommands.length; i++) {
							// preferring higher activities while sampling
							motorCommands[i] = rnd.nextDouble();
						}
						if(rnd.nextDouble() > 0.5) // activating only one turning in 50% of the cases
							motorCommands[2+rnd.nextInt(2)] = 0;
					}
				}
			}
			break;
		}
	}


}
