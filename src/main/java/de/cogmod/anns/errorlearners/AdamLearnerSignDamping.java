package de.cogmod.anns.errorlearners;

public class AdamLearnerSignDamping implements ErrorBasedLearner {

	private int numValues;
	private double learningRate = 0.001;
	private double beta1 = 0.9;
	private double beta1n1 = 0.1;
	private double beta2 =  0.999;
	private double beta2n1 = 0.001;
	private double beta3 =  0.9;
	private double beta3n1 = 0.1;
	private double beta1pt = 0;
	private double beta2pt = 0;
	private double beta3pt = 0;

	private double[] values;
	private double[] valueUpdates;
	private double[] nuValues;
	
	private double[] signBuffer;

	@SuppressWarnings("unused")
	private double tau = 1; // counting the number of updates... currently not used.

	public AdamLearnerSignDamping(final int numValues, final double learningRate) {
		this.numValues = numValues; 
		this.learningRate = learningRate; 

		this.values = new double[numValues];
		this.valueUpdates = new double[numValues];
		this.nuValues = new double[numValues];
		this.signBuffer = new double[numValues];
		
		this.beta1pt = this.beta1;
		this.beta2pt = this.beta2;
		this.beta3pt = this.beta3;
	}
	
	public AdamLearnerSignDamping(final int numValues, final double learningRate, final double beta1, final double beta2) {
		
		this(numValues, learningRate);

		this.beta1 = beta1;
		this.beta1n1 = 1.-this.beta1;
		this.beta2 = beta2; 
		this.beta2n1 = 1.-this.beta2;

		this.beta1pt = this.beta1;
		this.beta2pt = this.beta2;
	}

	@Override
	public void resetValues(final double[] newValues) {
		for(int i=0; i<numValues; i++) {
			this.values[i] = newValues[i];
			this.valueUpdates[i] = 0;
			this.nuValues[i] = 0;
			this.signBuffer[i] = 0;
		}
		tau = 1;
		this.beta1pt = this.beta1;
		this.beta2pt = this.beta2;
		this.beta3pt = this.beta3;
	}

	@Override
	public void shiftValues(int shiftSize) {
		for(int i=0; i<numValues-shiftSize; i++) {
			this.values[i] = this.values[i+shiftSize];
			this.signBuffer[i] = this.signBuffer[i+shiftSize];
			// some sets have suggested to me that a shift of the 
			// additional statistics is not beneficial... 
			// thus, setting them to zero seems better: 
			this.valueUpdates[i] = 0; //this.valueUpdates[i+shiftSize];
			this.nuValues[i] = 0; //this.nuValues[i+shiftSize];
		}
		for(int i=numValues-shiftSize; i<numValues; i++) {
			// keeping the former actual values in the values[] array.
//			this.signBuffer[i] = 0;
			this.valueUpdates[i] = 0;
			this.nuValues[i] = 0;
		}
		tau = 1;
		this.beta1pt = this.beta1;
		this.beta2pt = this.beta2;
//		this.beta3pt = this.beta3;
	}

	/**
	 * Update via Adam... 
	 * 
	 * NOTE: returned values array is used internally... value changes will result in faulty behavior!!!
	 */
	@Override
	public final double[] learningIteration(final double[] derivatives) {
		for(int i=0; i<this.numValues; i++) {
			this.nuValues[i] = this.beta2 * this.nuValues[i] + this.beta2n1 * derivatives[i] * derivatives[i];
			this.signBuffer[i] = this.beta3 * this.signBuffer[i] + this.beta3n1 * Math.signum(derivatives[i]);
			this.valueUpdates[i] = this.beta1 * this.valueUpdates[i] + this.beta1n1 * derivatives[i];
			this.values[i] -= this.learningRate * (Math.abs(signBuffer[i]) / (1.-beta3pt)) * (this.valueUpdates[i] / (1. - beta1pt)) 
					/ ( Math.sqrt(this.nuValues[i] / (1. - beta2pt)) + 1e-8);
		}		
		tau++;
		this.beta1pt *= this.beta1;
		this.beta2pt *= this.beta2;		
		this.beta3pt *= this.beta3;		
		return this.values;
	}

	/**
	 * Bounded update via Adam... 
	 * 
	 * NOTE: returned values array is used internally... value changes may result in faulty behavior!!!
	 */
	@Override
	public final double[] learningIterationBounded(final double[] derivatives, final double min, final double max) {
		double ombeta1tau = 1. - beta1pt;
		for(int i=0; i<this.numValues; i++) {
			this.nuValues[i] = this.beta2 * this.nuValues[i] + this.beta2n1 * derivatives[i] * derivatives[i];
			this.signBuffer[i] = this.beta3 * this.signBuffer[i] + this.beta3n1 * Math.signum(derivatives[i]);
			this.valueUpdates[i] = this.beta1 * valueUpdates[i] + this.beta1n1 * derivatives[i];
			double sqrtPlusValue = Math.sqrt(this.nuValues[i] / (1. - beta2pt)) + 1e-8;
			double change = this.learningRate * (Math.abs(signBuffer[i]) / (1.-beta3pt)) * (this.valueUpdates[i] / ombeta1tau) / sqrtPlusValue;
			if(this.values[i]-change < min || this.values[i]-change > max) {
				double diff = min - (this.values[i] - change);
				if(this.values[i]-change > max) {
					diff = max - (this.values[i] - change);
				}
				change -= diff;
				this.valueUpdates[i] = diff * ombeta1tau / (this.learningRate * (Math.abs(signBuffer[i]) / (1.-beta3pt))) * sqrtPlusValue;
			}
			this.values[i] -= change;
		}
		tau++;
		this.beta1pt *= this.beta1;
		this.beta2pt *= this.beta2;		
		this.beta3pt *= this.beta3;		
		return this.values;
	}

	/**
	 * Update via Adam... 
	 * 
	 * NOTE: returned values array is used internally... value changes may result in faulty behavior!!!
	 */
	@Override
	public final double[] learningIterationBounded(final double[] derivatives, final double[] minValues, final double[] maxValues) {
		double ombeta1tau = 1. - beta1pt;
		for(int i=0; i<this.numValues; i++) {
			this.nuValues[i] = this.beta2 * this.nuValues[i] + this.beta2n1 * derivatives[i] * derivatives[i];
			this.signBuffer[i] = this.beta3 * this.signBuffer[i] + this.beta3n1 * Math.signum(derivatives[i]);
			this.valueUpdates[i] = this.beta1 * valueUpdates[i] + this.beta1n1 * derivatives[i];
			double sqrtPlusValue = Math.sqrt(this.nuValues[i] / (1. - beta2pt)) + 1e-8;
			double change = this.learningRate * (Math.abs(signBuffer[i]) / (1.-beta3pt)) * (this.valueUpdates[i] / ombeta1tau) / sqrtPlusValue;
			if(this.values[i]-change < minValues[i] || this.values[i]-change > maxValues[i]) {
				double diff = minValues[i] - (this.values[i] - change);
				if(this.values[i]-change > maxValues[i]) {
					diff = maxValues[i] - (this.values[i] - change);
				}
				change -= diff;
				this.valueUpdates[i] = diff * ombeta1tau / (this.learningRate * (Math.abs(signBuffer[i]) / (1.-beta3pt))) * sqrtPlusValue;
			}
			this.values[i] -= change;
		}
		tau++;
		this.beta1pt *= this.beta1;
		this.beta2pt *= this.beta2;		
		this.beta3pt *= this.beta3;		
		return this.values;
	}

}
