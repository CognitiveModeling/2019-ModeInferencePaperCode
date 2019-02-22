/*******************************************************************************
 * JANNLab Neural Network Framework for Java
 * Copyright (C) 2012-2013 Sebastian Otte
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

package de.jannlab.math;


/**
 * This class provides some frequently used mathematical
 * helper methods.
 * <br></br>
 * @author Sebastian Otte
 */
public class MathTools {

	public static final double NULL_THRESHOLD_DOUBLE = 1.0e-9;
	public static final float  NULL_THRESHOLD_FLOAT  = 1.0e-9f;

	public static final double GOLDEN_RATIO = 0.5 * (1.0 + Math.sqrt(5.0));
	public static final double GOLDEN_RATIO_NORM  = 1.0 / GOLDEN_RATIO;
	public static final double GOLDEN_RATIO_NORMR = 1.0 - GOLDEN_RATIO_NORM;

	/**
	 * Checks, if the given value is approximately 0.0 based on the 
	 * threshold NULL_THRESHOLD_DOUBLE.
	 * <br></br>
	 * @param value The value which is to check.
	 * @return True if absolute of the given value is smaller than the threshold.
	 */
	public static boolean approxNull(final double value) {
		return Math.abs(value) < NULL_THRESHOLD_DOUBLE;
	}

	/**
	 * Checks, if the given value is approximately 0.0 based on the 
	 * threshold NULL_THRESHOLD_FLOAT.
	 * <br></br>
	 * @param value The value which is to check.
	 * @return True if absolute of the given value is smaller than the threshold.
	 */
	public static boolean approxNull(final float value) {
		return Math.abs(value) < NULL_THRESHOLD_FLOAT;
	}

	/**
	 * Checks, if the given value is approximately 0.0 based on the 
	 * threshold given by errorbound.
	 * <br></br>
	 * @param value The value which is to check.
	 * @param errorbound The explicitly given threshold.
	 * @return True if absolute of the given value is smaller than the threshold.
	 */
	public static boolean approxNull(final double value, final double errorbound) {
		return Math.abs(value) < Math.abs(errorbound);
	}

	/**
	 * Checks, if the given value is approximately 0.0 based on the 
	 * threshold given by errorbound.
	 * <br></br>
	 * @param value The value which is to check.
	 * @param errorbound The explicitly given threshold.
	 * @return True if absolute of the given value is smaller than the threshold.
	 */
	public static boolean approxNull(final float value, final float errorbound) {
		return Math.abs(value) < Math.abs(errorbound);
	}


	/**
	 * Computes the apprixmation of the exponential function e^{x} using
	 * the interpretation of IEEE-754 numbers. 
	 * <br></br>
	 * References:
	 * <br></br>
	 * Schraudolph, Nicol N.: A Fast, Compact Approximation of the 
	 * Exponential Function. In:Neural Computation11 (1998), S. 11???4
	 * <br></br>
	 * Optimized Exponential Functions for Java. 
	 * http://martin.ankerl.com/2007/02/11/optimized-exponential-
	 * functions-for-java/, 2007. ??? Visited on Januar, 12st 2012
	 * <br></br>
	 * @param value The argument of the exp function.
	 * @return Approximation of e^{x}.
	 */
	public static double fastExp(final double value) {
		final long tmp = (long)(1512775 * value) + 1072632447;
		return Double.longBitsToDouble(tmp << 32);
	}

	/**
	 * Computes a fast tanh function based on the approximation
	 * of the expontial function.
	 * <br></br>
	 * @param value The argument of the tanh function.
	 * @return Approximation of tanh(x).
	 */
	public static double fastTanh(final double value) {
		final double pos = fastExp(value);
		final double neg = fastExp(-value);
		return (pos - neg) / (pos + neg);
	}

	/**
	 * Return the index of the biggest values in an array of doubles.
	 * Returns -1 if an empty array is given.
	 * <br></br>
	 * @param args Double array.
	 * @return The index of the biggest value.
	 */
	public static int argmax(final double ...args) {
		if (args.length == 0) return -1;
		//
		int maxidx = 0;
		double max = args[0];
		for (int i = 1; i < args.length; i++) {
			if (args[i] > max) {
				max = args[i];
				maxidx = i;
			}
		}
		return maxidx;
	}

	/**
	 * Return the index of the smallest values in an array of doubles. 
	 * Returns -1 if an empty array is given.
	 * <br></br>
	 * @param args Double array.
	 * @return The index of the smallest value.
	 */
	public static int argmin(final double ...args) {
		if (args.length == 0) return -1;
		//
		int minidx = 0;
		double min = args[0];
		for (int i = 1; i < args.length; i++) {
			if (args[i] < min) {
				min = args[i];
				minidx = i;
			}
		}
		return minidx;
	}

	/**
	 * Returns the minimum value out of various values.
	 * If no values are given, POSITIVE_INFINITY will be returned. 
	 * @param args Double values.
	 * @return Minimum value of args.
	 */
	public static double min(final double ...args) {
		double min = Double.POSITIVE_INFINITY;
		//
		for (int i = 0; i < args.length; i++) {
			final double value = args[i];
			if (value < min) {
				min = value;
			}
		}
		//
		return min;
	}


	/**
	 * Returns value if value > 0 and 0 otherwise.
	 * @param value Int value.
	 * @return value or 0.
	 */
	public static int rect(final int value) {
		return (value > 0)?(value):(0);
	}

	/**
	 * Returns the minimum value out of various values.
	 * If no values are given, Integer.MAX_VALUE will be returned. 
	 * @param args Int values.
	 * @return Minimum value of args.
	 */
	public static int min(final int ...args) {
		int min = Integer.MAX_VALUE;
		//
		for (int i = 0; i < args.length; i++) {
			final int value = args[i];
			if (value < min) {
				min = value;
			}
		}
		//
		return min;
	}    


	/**
	 * Returns the maximum value out of various values.
	 * If no values are given, NEGATIVE_INFINITY will be returned. 
	 * @param args Double values.
	 * @return Maximum value of args.
	 */
	public static double max(final double ...args) {
		double max = Double.NEGATIVE_INFINITY;
		//
		for (int i = 0; i < args.length; i++) {
			final double value = args[i];
			if (value > max) {
				max = value;
			}
		}
		//
		return max;
	}

	/**
	 * Returns the maximum value out of various values.
	 * If no values are given, Integer.MIN_VALUE will be returned. 
	 * @param args Int values.
	 * @return Maximum value of args.
	 */
	public static int max(final int ...args) {
		int max = Integer.MIN_VALUE;
		//
		for (int i = 0; i < args.length; i++) {
			final int value = args[i];
			if (value > max) {
				max = value;
			}
		}
		//
		return max;
	}    


	/**
	 * Clamps a given value x returned as x' within the interval [lbd, ubd] 
	 * such that x' = lbd if x < lbd, x' = ubd if x > ubd or x' = x otherwise. 
	 * @param value The given value.
	 * @param lbd Lower bound of the clamp interval.
	 * @param ubd Upper bound of the clamp interval.
	 * @return A values within the interval [lbd, ubd]
	 */
	public static double clamp(
			final double value, final double lbd, final double ubd
			) {
		return (
				(value < lbd) ? (lbd) : (
						(value > ubd) ? (ubd) : (value)
						)
				);
	}

	public static void clamp(double[] values, final double lbd, final double ubd) {
		for(int i=0; i<values.length; i++) {
			values[i] = (values[i] < lbd) ? (lbd):
						( (values[i] > ubd) ? (ubd) : (values[i]) );
		}
	}


	public static double mean(
			final double[] data,
			final int offset,
			final int size
			) {
		double sum = 0;
		int    ptr = offset;
		//
		for (int i = 0; i < size; i++) {
			sum += data[ptr++];
		}
		//
		return (sum / (double)size);
	}

	public static double standardDeviation(
			final double[] data,
			final double mean,
			final int offset,
			final int size
			) {
		int    ptr = offset;
		double sum = 0;
		//
		for (int i = 0; i < size; i++) {
			final double x = data[ptr++];
			final double v = (x - mean);
			sum += (v * v);
		}
		//
		return Math.sqrt(
				(1.0 / (size - 1)) * sum
				);
	}

	public static double standardDeviation(
			final double[] data,
			final int offset,
			final int size
			) {
		int ptr = offset;
		//
		double sum1 = 0;
		double sum2 = 0;
		double mean = 0;
		//
		for (int i = 0; i < size; i++) {
			final double x = data[ptr++];
			mean += x;
			sum1 += (x * x);
			sum2 += (2 * x);
		}
		mean /= ((double)size);
		//
		final double sum = (
				sum1 - (mean * sum2) + (size * (mean * mean))
				);
		//
		return Math.sqrt(
				(1.0 / (size - 1)) * sum
				);
	}


	public static double smoothCrop(
			final double x, 
			final double min, 
			final double max,
			final double theta
			) {
		//
		assert(theta >= 0 && theta <= 1.0);
		//
		final double range    = (max - min);
		final double hrange   = 0.5 * range; 
		final double center   = 0.5 * (max + min);
		//
		// normalize x of [min, max] to [-1, 1]
		//
		final double nx     = (x - center) / hrange;
		final double sx     = Math.signum(nx);
		//
		if (Math.abs(nx) > theta) {
			//
			// return denormalized value.
			//
			final double stheta = sx * theta;
			final double itheta = 1.0 - theta;
			final double arg    = nx - (sx*theta);
			final double nvalue = stheta  + (itheta) * (Math.tanh(arg / itheta));
			//
			return (nvalue * hrange) + center;
		} else {
			return x;
		}
	}

	public static double interpolateLog(final double high, final double low, final double rel) {
		final double loghigh    = Math.log(high);
		final double loglow     = Math.log(low);
		final double loghighrel = loghigh * rel;
		final double loglowrel  = loglow * (1.0 - rel);
		final double exp        = Math.exp(loghighrel + loglowrel); 
		return exp;
	}

}
