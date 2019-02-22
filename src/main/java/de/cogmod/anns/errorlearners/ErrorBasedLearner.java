package de.cogmod.anns.errorlearners;

public interface ErrorBasedLearner {

	public enum WEIGHT_UPDATE{
		GradientDescent,
		Adam,
		AdamLearnerSignDamping
	};


	/**
	 * Resets the numerical values to the new values.
	 * Assumes a totally new learning run for now.
	 * 
	 * @param newValues
	 */
	public void resetValues(final double[] newValues);
	
	/**
	 * Shifts all the values by the specified shift size to the front. 
	 * Copies the last shiftSize values to the missing values.
	 * 
	 * @param shiftSize
	 */
	public void shiftValues(int shiftSize);

	public double[] learningIteration(double[] derivatives); 
	
	public double[] learningIterationBounded(final double[] derivatives, final double min, final double max);
	
	public double[] learningIterationBounded(final double[] derivatives, final double[] minValues, final double[] maxValues);
}
