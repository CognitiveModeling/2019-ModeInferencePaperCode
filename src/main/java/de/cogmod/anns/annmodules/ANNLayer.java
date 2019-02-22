package de.cogmod.anns.annmodules;

import java.util.Random;

import de.cogmod.anns.errorlearners.ErrorBasedLearner;
import de.cogmod.anns.errorlearners.ErrorBasedLearner.WEIGHT_UPDATE;
import de.cogmod.utilities.LearningListener;

public interface ANNLayer {
	
	public enum LayerTypes{
		RNN,MLP,LSTM,LSTMresevoir,RNNin,LSTMsplitInput
	};
	
	/**
	 * Returns a newly created array of output activities. 
	 * NOTE: returned array may be used and values may be overwritten!!!
	 * 
	 * @param timePoint
	 * @return
	 */
	public abstract double[] getOutputActivities(int timePoint);
	
	/**
	 * Returns a newly created array of the recorded input BW signals at the specified time point.
	 * NOTE: returned array may be used and values may be overwritten!!!
	 *
	 * @param timePoint
	 * @return
	 */
	public abstract double[] getInputBW(int timePoint);

	/**
	 * Sets the bw signals of the output neurons of this layer to the specified values. 
	 * 
	 * @param bwSignals
	 * @param timePoint
	 */
	public abstract void setOutputBW(double[] bwSignals, int timePoint);

	/**
	 * Rebuffers this layer (resets all) to the specified length. 
	 * 
	 * @param sequencelength
	 */
	public void rebufferOnDemand(final int sequencelength);

	/**
	 * Resets all (buffered and current) network activities, error values etc. to zero.
	 */
	public void resetAllActivitiesToZero();

	/**
	 * Resets only the first time step activities to zero. 
	 */
	public void resetFirstTimeStepActivitiesToZero();
	
	/**
	 * Executes one forward pass through this network at the specified point in time 
	 * given the specified input. 
	 * 
	 * @param input the input into this layer at the current point in time
	 * @param t point in time
	 * @param prevt the previous point in time (may be wrapped around the buffer); 
	 * 			set to -1 if this is an initial forward pass step (algorithm then ignores internal recurrences).
	 */
	public void oneForwardPassAt(double[] input, int t, int prevt);

	/**
	 * Computes one backward pass step through the network at time index t. 
	 * Surrounded by the previous time index and next time index.
	 * Assumes that the bw signals in the output layer are set (starts with them).
	 * 
	 * @param target the target values at this time index - might be null, indicating that no target values are set here
	 * @param t the point in time
	 * @param prevt previous time index (usually t-1) - -1 indicates that t is the first time index in a forward pass series
	 * @param nextt next time index (usually t+1) - -1 indicates that t is the last time index in a forward pass series
	 */
	public void oneBackwardPassAt(int t, int prevt, int nextt);
	
	/**
	 * Initializes the weights randomly and normally distributed with
	 * the specified standard deviation. 
	 * 
	 * @param rnd Instance of Random.
	 * @param stddev 
	 */
	public void initializeWeights(final Random rnd, final double stddev);

	/**
	 * Returns the number of adaptable weights in this layer. 
	 * 
	 * @return
	 */
	public int getNumWeights();

	/**
	 * Reads the own current weight values (i.e. the current weights of the neural layer)
	 * and writes them systematically into the provided array. 
	 * ASSUMES that the provided myWeights array has appropriate length. 
	 * 
	 * @param myWeights
	 */
	public void readWeights(double[] myWeights);

	/**
	 * Writes the provided weight values systematically into its own current weights.
	 * ASSUMES that the provided array has appropriate length.
	 * 
	 * @param myNewWeights
	 */
	public void writeWeights(double[] myNewWeights);
	
	/**
	 * Reads the currently recorded error signals in the weights of this layer and writes them systematically
	 * into the slots of the provided array.  
	 * ASSUMES that the provided array has appropriate length.
	 * 
	 * @param myDiffWeights
	 */
	public void readDiffWeights(double[] myDiffWeights);
	
	/**
	 * Accumulates the weight-specific error signals for backpropagation over
	 * the specified time window, such that 
	 * proper error-based weight adaptation can be executed. 
	 * 
	 * @param tInit start time point
	 * @param tLast end time point 
	 */
	public void setWeightsDerivativesGeneral(int tInit, int tLast);

	/**
	 * The number of hidden activities that are relevant for initializing the hidden state of an ANN layer
	 * (given it is recurrent, 0 otherwise).
	 * @return number of hidden states that determine the unfolding activity of this layer. 
	 */
	public int getHiddenActivitiesNum();

	/**
	 * Returns the individual neural activity bounds of the hidden states of each neuron.
	 * Assumes that provided arrays are sufficiently large. (>= getHiddenActivitiesNum() )
	 * 
	 * @param lowerBounds 
	 * @param upperBounds
	 */
	public void getInternalActivityBounds(double[] lowerBounds,  double[] upperBounds);
	
	/**
	 * Reads the bwErrors that are relevant to adapt the internal state activities of this layer
	 * 
	 * @param myCurrentBWErrors array of sufficient size to fill it with the relevant bw errors. 
	 * @param timePoint at which buffer point to read from
	 */
	public void readBWErrors(double[] myCurrentBWErrors, int timePoint);

	/**
	 * Overwrites the activities of those hidden states of this layer that are relevant for 
	 * recurrent activations. 
	 * 
	 * @param activities array of sufficient size to write all hidden neural state activities. 
	 * @param timePoint at which buffer point to write these activities.
	 */
	public void writeInternalActivities(double[] activities, int timePoint);

	public void readInternalActivities(double[] activities, int timePoint);

	
	/**
	 * Returns the output size of this layer. 
	 * @return
	 */
	public int getOutputSize();

	/**
	 * Returns the input size of this layer. 
	 * @return
	 */
	public int getInputSize();

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
			);
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
			double momentum,
			ErrorBasedLearner.WEIGHT_UPDATE activeInferenceUpdateType
			);
	/**
	 * Computes the forward pass, i.e., propagates an input 
	 * vector through the network to the output layer. 
	 * This method does so for one time step online... 
	 * .. assuming the recurrence unfolds in the first time step
	 * 
	 * @param input Input vector.
	 * @return Output vector.
	 */
	public double[] forwardPass(final double[] input);
	
	public void writeNetwork(String fileName);

}
