package de.cogmod.utilities;

import java.util.Locale;
import java.util.Random;

/**
 * @author  Martin Butz
 */
public class Tools {

	public static void shuffle(
			final int[] data, 
			final Random rnd
			) {
		//
		final int size = data.length;
		//
		for (int i = size; i > 1; i--) {
			final int ii = i - 1;
			final int r  = rnd.nextInt(i);
			//
			final int temp = data[ii];
			data[ii] = data[r];
			data[r] = temp;
		}
	}


	public static double[] createFilledArray(int length, double value) {
		double[] ret = new double[length];
		for(int i=0; i<length; i++)
			ret[i] = value;
		return ret;
	}
	
	public static void fillArray(double[] arr, double value) {
		for(int i=0; i<arr.length; i++)
			arr[i] = value;
	}

	public static void setValArray(double[] arr, int startIndex, int pastEndIndex, double value) {
		for(int i=startIndex; i<pastEndIndex; i++)
			arr[i] = value;		
	}
	
	/**
	 * Copies the values in newValues into the corresponding positions in theArray.
	 * @param newValues
	 * @param theArray
	 */
	public static void copyToArray(double[] newValues, double[] theArray) {
		for(int i=0; i<newValues.length; i++)
			theArray[i] = newValues[i];
	}

	/**
	 * Copies the values in newValues into the corresponding positions in theArray.
	 * @param newValues
	 * @param sourceArrayStartIndex where in the addition array to start from (until the end of addition array).
	 * @param theArray
	 */
	public static void copyToArray(double[] newValues, int sourceStartIndex, double[] theArray) {
		for(int i=sourceStartIndex,j=0; i<newValues.length; i++,j++)
			theArray[j] = newValues[i];
	}

	/**
	 * Copies the values in newValues into the corresponding positions in theArray.
	 * @param newValues
	 * @param theArray
	 * @param destArrayStartIndex where in the destination array to start from (copying full source index beginning at this index in the destination array).
	 */
	public static void copyToArray(double[] newValues, double[] theArray, int destArrayStartIndex) {
		for(int i=0, j=destArrayStartIndex; i<newValues.length; i++,j++)
			theArray[j] = newValues[i];
	}

	/**
	 * Copies the values in newValues into the corresponding positions in theArray.
	 * @param newValues
	 * @param sourceArrayStartIndex where in the addition array to start from (until the end of addition array).
	 * @param destStartIndex where in the destination array to start from (copying full source index beginning at this index in the destination array).
	 * @param theArray
	 */
	public static void copyToArray(double[] newValues, int sourceArrayStartIndex, double[] theArray, int destArrayStartIndex) {
		for(int i=sourceArrayStartIndex, j=destArrayStartIndex; i<newValues.length; i++,j++)
			theArray[j] = newValues[i];
	}

	public static void copyToArray(double[] newValues, int sourceArrayStartIndex, int sourceArrayPastEndIndex, double[] theArray, int destArrayStartIndex) {
		for(int i=sourceArrayStartIndex, j=destArrayStartIndex; i<sourceArrayPastEndIndex; i++,j++)
			theArray[j] = newValues[i];
	}

	public static void multiplyArray(double mul, double[] theArray) {
		for(int i=0; i<theArray.length; i++)
			theArray[i] *= mul;
	}

	public static void multiplyArray(double mul, double[][] theArray) {
		for(int i=0; i<theArray.length; i++)
			for(int j=0; j<theArray[i].length; j++)
				theArray[i][j] *= mul;
	}
	
	public static void addToArray(double value, double[][] theArray) {
		for(int i=0; i<theArray.length; i++)
			for(int j=0; j<theArray[i].length; j++)
				theArray[i][j] += value;
	}
	
	public static void dotDivision(double[][] values, double[][] divisions) {
		for(int i=0; i<values.length; i++)
			for(int j=0; j<values[i].length; j++)
				values[i][j] /= divisions[i][j];
	}
	
	public static void dotMultiplication(double[][] values, double[][] divisions) {
		for(int i=0; i<values.length; i++)
			for(int j=0; j<values[i].length; j++)
				values[i][j] *= divisions[i][j];
	}
	
	
	/**
	 * Adds the values in addition to the corresponding values in theArray.
	 * @param addition
	 * @param theArray
	 */
	public static void addToArray(double[][] addition, double[][] theArray) {
		for(int i=0; i<addition.length; i++)
			for(int j=0; j<addition[i].length; j++)
				theArray[i][j] += addition[i][j];
	}
	
	/**
	 * Adds the values in addition to the corresponding values in theArray.
	 * @param addition
	 * @param theArray
	 */
	public static void addToArray(double[] addition, double[] theArray) {
		for(int i=0; i<addition.length; i++)
			theArray[i] += addition[i];
	}


	/**
	 * Adds the values in addition to the corresponding values in theArray.
	 * @param addition
	 * @param theArray
	 */
	public static void addToArray(double[] addition, int sourceArrayStartIndex, double[] theArray) {
		for(int i=sourceArrayStartIndex, j=0; i<addition.length; i++,j++)
			theArray[j] += addition[i];
	}


	/**
	 * Adds the values in addition to the corresponding values in theArray.
	 * @param addition
	 * @param theArray
	 */
	public static void addToArray(double[] addition, int sourceArrayStartIndex, int sourceArrayEndPoint, double[] theArray) {
		for(int i=sourceArrayStartIndex, j=0; i<sourceArrayEndPoint; i++,j++)
			theArray[j] += addition[i];
	}

	/**
	 * Returns the signed minimum of the positive value max and the absolute value of "value"
	 * Thus, this method returns a value that lies within the interval [-max;max]
	 * 
	 * @param max an assumed positive value that is the maximal absolute value.
	 * @param value
	 * @return the resulting signed minimum of the two values.
	 */
	public static double minAbs(double max, double value) {
		if(value > max)
			return max;
		if(value < -max)
			return -max;
		return value;
	}

	/**
	 * Returns an array of the signed minia of the positive value max and the absolute value of "value"
	 * Thus, this method returns a value that lies within the interval [-max;max]
	 * 
	 * @param max an assumed positive value that is the maximal absolute value.
	 * @param value
	 * @return the resulting signed minimum of the two values.
	 */
	public static void minAbs(double max, double[] values) {
		for(int i=0; i<values.length; i++) {
			if(values[i] > max)
				values[i] = max;
			else if(values[i] < -max)
				values[i] = -max;
		}
	}

	public static double getMinValueColumn(double[][] arr, int columnIndex) {
		double min = Double.MAX_VALUE;
		for(int r=0; r<arr.length; r++) {
			if(arr[r][columnIndex] < min)
				min = arr[r][columnIndex];
		}
		return min;
	}
	
	public static double getMaxValueColumn(double[][] arr, int columnIndex) {
		double max = -Double.MAX_VALUE;
		for(int r=0; r<arr.length; r++) {
			if(arr[r][columnIndex] > max)
				max = arr[r][columnIndex];
		}
		return max;
	}

	public static String toStringArray(double[] arr) {
		StringBuffer ret = new StringBuffer("[");
		for(int i=0; i<arr.length; i++) {
			ret.append(String.format(Locale.US, "%.1f", arr[i]));
			if(i+1 < arr.length)
				ret.append(", ");
		}
		ret.append("]");
		return ret.toString();
	}

	public static String toStringArray(int[] arr) {
		StringBuffer ret = new StringBuffer("[");
		for(int i=0; i<arr.length; i++) {
			ret.append(arr[i]);
			if(i+1 < arr.length)
				ret.append(", ");
		}
		ret.append("]");
		return ret.toString();
	}

	public static String toStringArray(double[] arr, int length) {
		return toStringArrayPrecision(arr, length, 1);
	}

	public static String toStringArrayPrecision(double[] arr, int precision) {
		return toStringArrayPrecision(arr, arr.length, precision);
	}

	public static String toStringArrayPrecision(double[] arr, int length, int precision) {
		StringBuffer ret = new StringBuffer("[");
		for(int i=0; i<arr.length && i<length; i++) {
			ret.append(String.format(Locale.US, "%."+precision+"f", arr[i]));
			if(i+1 < arr.length && i+1 < length)
				ret.append(", ");
		}
		ret.append("]");
		return ret.toString();
	}

	public static String toStringArray(double[][] arr) {
		return toStringArrayPrecision(arr, 1);
	}

	public static String toStringArrayPrecision(double[][] arr, int precision) {
		StringBuffer ret = new StringBuffer("[");
		for(int i=0; i<arr.length; i++) {
			ret.append("[");
			for(int j=0; j<arr[i].length; j++) {
				ret.append(String.format(Locale.US, "%."+precision+"f", arr[i][j]));
				if(j+1 < arr[i].length)
					ret.append(", ");
			}
			ret.append("]\n");
			if(i+1 < arr.length)
				ret.append(", ");
		}
		ret.append("]");
		return ret.toString();
	}

	public static String toStringArray(double[][] arr, int length1, int length2) {
		return toStringArrayPrecision(arr, length1, length2, 1);
	}


	public static String toStringArrayPrecision(double[][] arr, int length1, int length2, int precision) {
		StringBuffer ret = new StringBuffer("[");
		for(int i=0; i<arr.length && i<length1; i++) {
			ret.append("[");
			for(int j=0; j<arr[i].length && j<length2; j++) {
				ret.append(String.format(Locale.US, "%."+precision+"f", arr[i][j]));
				if(j+1 < arr[i].length && j+1<length2)
					ret.append(", ");
			}
			ret.append("]");
			if(i+1 < arr.length && i+1 < length1)
				ret.append(", ");
		}
		ret.append("]");
		return ret.toString();
	}

	public static boolean equalsArray(double[] arr1, double[] arr2) {
		if(arr1==null) {
			if(arr2==null)
				return true;
			return false;
		}
		if(arr2==null)
			return false;
		if(arr1.length != arr2.length)
			return false;
		for(int i=0; i<arr1.length; i++)
			if(arr1[i]!=arr2[i])
				return false;
		return true;
	}

	//Dania begin
	/**
	 * Computes the Eucledian distance between the current output and
	 * a given target vector.
	 */
	public static double EucDist(final double[] output, final double[] target) {
		double dist = 0;
		//
		for (int i = 0; i < target.length; i++) {
			final double d = output[i] - target[i];
			dist += (d * d);
		}
		//
		return Math.sqrt(dist);
	}
	//Dania End
	
	/**
	 * Computes the RMSE of the current output and
	 * a given target vector.
	 * @param target Target vector.
	 * @return RMSE value.
	 */
	public static double RMSE(final double[] output, final double[] target) {
		//
		assert(output.length > 0);
		assert(target.length > 0);
		assert(target.length <= output.length);
		//
		double error = 0;
		//
		for (int i = 0; i < target.length; i++) {
			final double e = output[i] - target[i];
			error += (e * e);
		}
		//
		return Math.sqrt(error / (double)(target.length));
	}


	/**
	 * Computes the RMSE of the current output and
	 * a given target vector.
	 * @param target Target vector.
	 * @return RMSE value.
	 */
	public static double RMSE(final double[][] output, final double[][] target) {
		//
		final int length = Math.min(output.length, target.length);
		//
		double error = 0;
		int    ctr   = 0;
		//
		for (int t = 0; t < length; t++) {
			assert(output[t].length > 0);
			assert(target[t].length > 0);
			assert(target[t].length == output[t].length);
			//
			for (int i = 0; i < target[t].length; i++) {
				final double e = output[t][i] - target[t][i];
				error += (e * e);
				ctr++;
			}
		}
		//
		return Math.sqrt(error / (double)(ctr));
	}

	/**
	 * Computes the MSE of the current output and
	 * a given target vector.
	 * @param target Target vector.
	 * @return MSE value.
	 */
	public static double MSE(final double[][] output, final double[][] target) {
		//
		final int length = Math.min(output.length, target.length);
		//
		double error = 0;
		int    ctr   = 0;
		//
		for (int t = 0; t < length; t++) {
			assert(output[t].length > 0);
			assert(target[t].length > 0);
			assert(target[t].length == output[t].length);
			//
			for (int i = 0; i < target[t].length; i++) {
				final double e = output[t][i] - target[t][i];
				error += (e * e);
				ctr++;
			}
		}
		//
		return error / (double)(ctr);
	}


	public static double meanSquaredError(double[] a, double[] b) {
		assert(a.length==b.length);

		double err = 0;
		for(int i=0; i<a.length; i++) {
			double diff = a[i] - b[i];
			err += diff * diff;
		}
		return err/a.length;
	}


	public static void map(final double[] from1, final double[] from2, final double[] to) {
		int idx = 0;
		for (int l1 = 0; l1 < from1.length; l1++) {
			to[idx++] = from1[l1];
		}
		for (int l2 = 0; l2 < from2.length; l2++) {
			to[idx++] = from2[l2];
		}
	}

	public static void map(final double[] from1, final double[] from2, final double[] from3, final double[] to) {
		int idx = 0;
		for (int l1 = 0; l1 < from1.length; l1++) {
			to[idx++] = from1[l1];
		}
		for (int l2 = 0; l2 < from2.length; l2++) {
			to[idx++] = from2[l2];
		}
		for (int l3 = 0; l3 < from3.length; l3++) {
			to[idx++] = from3[l3];
		}
	}

	public static void map(final double[] from1, final double[] from2, final double[] from3, final double[] from4, final double[] to) {
		int idx = 0;
		for (int l1 = 0; l1 < from1.length; l1++) {
			to[idx++] = from1[l1];
		}
		for (int l2 = 0; l2 < from2.length; l2++) {
			to[idx++] = from2[l2];
		}
		for (int l3 = 0; l3 < from3.length; l3++) {
			to[idx++] = from3[l3];
		}
		for (int l4 = 0; l4 < from4.length; l4++) {
			to[idx++] = from4[l4];
		}
	}

	public static void map(final double[] from, final double[][][][] to) {
		int idx = 0;
		for (int l1 = 0; l1 < to.length; l1++) {
			for (int l2 = 0; l2 < to[l1].length; l2++) {
				double[][] wll = to[l1][l2];
				if (wll != null) {
					for (int i = 0; i < wll.length; i++) {
						for (int j = 0; j < wll[i].length; j++) {
							wll[i][j] = from[idx++];
						}
					}
				}
			}
		}
	}

	public static void map(final double[][][][] from, final double[] to) {
		int idx = 0;
		for (int l1 = 0; l1 < from.length; l1++) {
			for (int l2 = 0; l2 < from[l1].length; l2++) {
				double[][] wll = from[l1][l2];
				if (wll != null) {
					for (int i = 0; i < wll.length; i++) {
						for (int j = 0; j < wll[i].length; j++) {
							to[idx++] = wll[i][j];
						}
					}
				}
			}
		}
	}


	public static void map(final double[] from, final double[][] to) {
		int idx = 0;
		for (int l1 = 0; l1 < to.length; l1++) {
			for (int l2 = 0; l2 < to[l1].length; l2++) {
				to[l1][l2] = from[idx++];
			}
		}
	}

	public static void map(final double[][] from, final double[] to) {
		int idx = 0;
		for (int l1 = 0; l1 < from.length; l1++) {
			for (int l2 = 0; l2 < from[l1].length; l2++) {
				to[idx++] = from[l1][l2];
			}
		}
	}

	public static void bindArray(final double[]arr, double min, double max) {
		for(int i=0; i<arr.length; i++) {
			if(arr[i]<min) {
				arr[i]=min;
			}else if(arr[i] > max) {
				arr[i] = max;
			}
		}
	}
	
	public static void normalizeVectorLength(final double[] arr, double vectorLength) {
		double val = 1e-10;
		for(int i=0; i<arr.length; i++) {
			val += arr[i] * arr[i];
		}
		val = Math.sqrt(val);
		for(int i=0; i<arr.length; i++) {
			arr[i] /= val;
		}		
	}

	public static void enforceProbabilityMass(final double[] arr) {
		double val = 1e-10;
		for(int i=0; i<arr.length; i++) {
			val += arr[i];
		}
		for(int i=0; i<arr.length; i++) {
			arr[i] /= val;
		}		
	}

	public static void enforceProbabilityMass(final double[] arr, int startIndex, int pastLastIndex) {
		double val = 1e-10;
		for(int i=startIndex; i<pastLastIndex; i++) {
			val += arr[i];
		}
		for(int i=startIndex; i<pastLastIndex; i++) {
			arr[i] /= val;
		}		
	}


}