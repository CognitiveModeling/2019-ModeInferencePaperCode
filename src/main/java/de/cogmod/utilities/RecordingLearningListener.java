package de.cogmod.utilities;

import java.util.LinkedList;

/**
 * @author Sebastian Otte & Martin V. Butz
 */
public class RecordingLearningListener implements LearningListener {
	
	private int triggerFrequency;
	private double[] errorSum;
	private LinkedList<double[]> recordedErrors;
	private int numValues;	
	
	public RecordingLearningListener(int numValues, int triggerFrequency) {
		this.triggerFrequency = triggerFrequency;
		recordedErrors = new LinkedList<double[]>();
		this.numValues = numValues;
		errorSum = new double[numValues];
	}

	@Override
    public void afterEpoch(final int epoch, final double... trainingerror) {
		for(int i=0; i<trainingerror.length; i++) 
			errorSum[i] += trainingerror[i];
        if(epoch % this.triggerFrequency == 0) {
//        	System.out.println("epoch: " + epoch + " av.err.: " + (errorSum/this.triggerFrequency) + "curr.err.: " + trainingerror);
    		for(int i=0; i<errorSum.length; i++) 
    			errorSum[i] /= this.triggerFrequency;
    		recordedErrors.add(errorSum.clone());
    		for(int i=0; i<errorSum.length; i++) 
    			errorSum[i] = 0;
        }
    }
	
	public double[][] getRecordedErrors() {
		double[][] retArray = new double[recordedErrors.size()][numValues];
		recordedErrors.toArray(retArray);
		return retArray;
	}
}