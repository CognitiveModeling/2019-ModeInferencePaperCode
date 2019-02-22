package de.cogmod.problems;

public interface OutputToInputMapping {
	/**
	 * Maps the sensor output given the previous sensor input to the next sensor input.
	 * Typically used to indicate if the output-> next input is an identity mapping or 
	 * an additive mapping (when the output is a predicted sensory state change).
	 * 
	 * @param sensorInput Array that contains in the first values the sensor input readings (may have additional values).
	 * @param sensorOutput Array that contains in the first values the sensor output readings (may have additional values).
	 * @param nextSensorInput
	 */
	public void mapStateAndOutputToNextInput(double[] sensorInput, double[] sensorOutput, double[] nextSensorInput);
}
