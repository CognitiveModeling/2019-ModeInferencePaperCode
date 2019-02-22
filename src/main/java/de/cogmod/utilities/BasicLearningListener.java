package de.cogmod.utilities;

/**
 * @author Sebastian Otte & Martin Butz
 */
public class BasicLearningListener implements LearningListener {
	
	private int triggerFrequency = 100;
	private double[] errorSum = null;
	
	public BasicLearningListener() {
	}
	
	public BasicLearningListener(int triggerFrequency) {
		this.triggerFrequency = triggerFrequency;
	}

	@Override
    public void afterEpoch(final int epoch, final double... trainingerror) {
		if(errorSum==null)
			errorSum = new double[trainingerror.length];
		for(int i=0; i<errorSum.length; i++)
			errorSum[i] += trainingerror[i];
        if(epoch % this.triggerFrequency==0) {
        	System.out.print("epoch: " + epoch);
    		for(int i=0; i<errorSum.length; i++) {
    			System.out.print(" av.err." + i + ": " + (errorSum[i]/this.triggerFrequency) + "curr.err.: " + trainingerror[i]);
    			errorSum[i] = 0;
    		}
    		System.out.println();
        }
    }
}