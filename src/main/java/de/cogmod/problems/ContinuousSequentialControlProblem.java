package de.cogmod.problems;

import javax.swing.JFrame;
import javax.swing.JPanel;

public interface ContinuousSequentialControlProblem {

	public String getProblemName();
	
	/**
	 * Returns the sizes of the readings of the different sensors available. 
	 * 
	 * @return array length specifies the different types of sensor (i.e. state) information available, individual entries
	 * specify the size of the individual sensor reading vectors.
	 */
	public int getSensorInputSize();
	
	
	/**
	 * Returns the sizes of the readings of the different sensors available. 
	 * 
	 * @return array length specifies the different types of sensor (i.e. state) information available, individual entries
	 * specify the size of the individual sensor reading vectors.
	 */
	public int getSensorOutputSize();
	
	/**
	 * Returns the motor command size, i.e., the size of the motor commands that can be invoked.
	 * 
	 * @return
	 */
	public int getMotorSize();
	
	/**
	 * Returns an array of the min=0/max=1 allowed motor values. 
	 * Thus, the returned array is of size [2][getMotorSize()]
	 * @return
	 */
	public double[][] getMinMaxMotorValues();
	
	/**
	 * Executes the specified motor commands and sets the sensor readings according to the 
	 * problem state after executing the motor commands. 
	 * 
	 * @param motorCommands
	 * @param nextSensorReadings
	 */
	public void executeAndGet(double[] motorCommands, double[] sensorOutput, double[] nextSensorInput);

	/**
	 * Resets the control problem to a start state.
	 */
	public void reset();
	
	/**
	 * Potentially a visualization of the control problem environment. 
	 * 
	 * @return the panel visualization - null if not supported by the problem instance.
	 */
	public JPanel getPanel();
	
	/**
	 * Enables user interactions with the game visualization that is assumed to be contained in the 
	 * provided frame. 
	 * 
	 * @param frame the frame in which the JPanel of the game can be found .- or rather, through which the 
	 * 				game functionality may be controlled by the user. 
	 */
	public void activateKeyListener(JFrame frame);
	
	/**
	 * Sets if currently draw updates should be triggered or not.
	 * 
	 * @param doDraw if drawing updates occur after each motor invocation.
	 */
	public void setDoDraw(boolean doDraw);
	
	/**
	 * Sets a target vector.
	 * @param target
	 */
	public void setCurrentTarget(double[] target);

}
