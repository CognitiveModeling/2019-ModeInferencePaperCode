package de.cogmod.anns.errorlearners;

public class GradientDescentLearner implements ErrorBasedLearner {

	private int numValues;
	private double learningRate;
	private double momentumRate;
	
	private double[] values;
	private double[] valueUpdates;
	
	public GradientDescentLearner(final int numValues, final double learningRate, final double momentumRate) {
		this.numValues = numValues; 
		this.learningRate = learningRate; 
		this.momentumRate = momentumRate;
		
		this.values = new double[numValues];
		this.valueUpdates = new double[numValues];
	}
	
	@Override
	public void resetValues(final double[] newValues) {
		for(int i=0; i<numValues; i++) {
			this.values[i] = newValues[i];
			this.valueUpdates[i] = 0;
		}
	}
	
	@Override
	public void shiftValues(int shiftSize) {
		for(int i=0; i<numValues-shiftSize; i++) {
			this.values[i] = this.values[i+shiftSize];
			this.values[i] = this.values[i+shiftSize];
			// some sets have suggested to me that a shift of the 
			// additional statistics is not beneficial... 
			// thus, setting them to zero seems better: 
			this.valueUpdates[i] = 0;//this.valueUpdates[i+shiftSize];
		}
		for(int i=numValues-shiftSize; i<numValues; i++) {
			this.valueUpdates[i] = 0;
		}
	}
	
	@Override
	public final double[] learningIteration(final double[] derivatives) {
		for(int i=0; i<numValues; i++) {
			this.valueUpdates[i] = this.learningRate * derivatives[i] +
					this.momentumRate * valueUpdates[i];
			this.values[i] -= this.valueUpdates[i];
		}
		return this.values;
	}

	@Override
	public final double[] learningIterationBounded(final double[] derivatives, final double min, final double max) {
		for(int i=0; i<numValues; i++) {
			double change = this.learningRate * derivatives[i] +
					this.momentumRate * valueUpdates[i];
			if(this.values[i]-change < min || this.values[i]-change > max) {
				double diff = min - (this.values[i] - change);
				if(this.values[i]-change > max) {
					diff = max - (this.values[i] - change);
				}
				change -= diff;
			}
			this.valueUpdates[i] = change;
			this.values[i] -= this.valueUpdates[i];
		}
		return this.values;
	}

	@Override
	public double[] learningIterationBounded(final double[] derivatives, final double[] minValues, final double[] maxValues) {
		for(int i=0; i<this.numValues; i++) {
			double change = this.learningRate * derivatives[i] +
					this.momentumRate * valueUpdates[i];
			if(this.values[i]-change < minValues[i] || this.values[i]-change > maxValues[i]) {
				double diff = minValues[i] - (this.values[i] - change);
				if(this.values[i] - change > maxValues[i]) {
					diff = maxValues[i] - (this.values[i] - change);
				}
				change -= diff;
			}
			this.valueUpdates[i] = change;
			this.values[i] -= this.valueUpdates[i];
		}
		return this.values;		
	}
}
