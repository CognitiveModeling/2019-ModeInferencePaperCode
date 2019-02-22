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

package de.jannlab.examples.tools;

import de.jannlab.examples.tools.Vector3d;

/**
 * This class is a vector of two double values x, y.
 * It contains some static methods for operating in R2 (2 dimensional space).
 * <br></br>
 * @author Sebastian Otte / Martin V. Butz
 *
 */
public class Vector2d {
	
    public double x = 0.0d;
	public double y = 0.0d;
	
	/**
	 * Create an instance of Vector2f with zero values.
	 */
	public Vector2d() {
		//
	}
	
	/**
	 * Creates in instance of Vector2f by a given Vector3f.
	 * The x and y value will by divided by the z-value.  
	 * @param v Instance of Vector3f.
	 */
	public Vector2d(final Vector3d v) {
	    this(v.x / v.z, v.y / v.z);
	}
	
    /**
     * Normalizes a vector and stores the result that is a vector
     * in the same direction with length of 1 in pvret.
     * @param pv Source vector.
     * @param pvret Normalized vector.
     */
    public static void normalize(Vector2d pv, Vector2d pvret) {
        double l = 1.0f / pv.length();
        pvret.x = pv.x * l;
        pvret.y = pv.y * l;
    }
    /**
     * Normalizes a vector and returns the result that is a vector
     * in the same direction with length of 1 as a new vector.
     * @param pv The source vector.
     * @return A new normalized vector.
     */
    public static Vector2d normalize(Vector2d pv) {
        double l = 1.0f / pv.length();
        return new Vector2d(pv.x * l, pv.y * l);
    }
    
    /**
     * Creates an instance of Vector2f for a given x and y value.
     * <br></br>
     * @param px The x value.
     * @param py The y value.
     */
	public Vector2d(double px, double py) {
		this.x = px;
		this.y = py;
	}
	
    /**
     * Creates an instance of Vector2f by a given Vector2f.
     * <br></br>
     * @param pv An Instance of Vector2f.
     */
	public Vector2d(Vector2d pv) {
		this.x = pv.x;
		this.y = pv.y;
	}
	
    /**
     * Copy the values of a given Vector2f.
     * <br></br>
     * @param pv An instance of Vector2f.
     */	
	public void copy(Vector2d pv) {
		this.x = pv.x;
		this.y = pv.y;
	}
	
	/**
	 * Creates a copy this instance.
	 * <br></br>
	 * @return Copy of this instance.
	 */
	public Vector2d copy() {
		return new Vector2d(this.x, this.y);
	}
	
	/**
	 * Copies the values of a given double array.
	 * @param pd Values as double[].
	 */
	public void copy(double[] pd) {
		this.x = pd[0];
		this.y = pd[1];
	}
	
	/**
	 * Copies the values of a given double array at an specific index.
	 * @param pd Values as double[].
	 * @param poffset The values offset.
	 */
	public void copy(double[] pd, int poffset) {
		this.x = pd[poffset];
		this.y = pd[poffset + 1];
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public String toString() {
		return "Vector2f(" + this.x + "," + this.y + ")";
	}
	
	/**
	 * Computes the length of the vector (euclidian).
	 * @return Vector length.
	 */
	public double length() {
		return (double)Math.sqrt((this.x * this.x) + (this.y * this.y));
	}

	/**
     * Computes the squared length of the vector (euclidian). Can be 
     * used as a faster metric then length, of only the relative order
     * of the vectors is needed. 
     * @return Vector length without final sqrt.
     */
	public double length2() {
		return ((this.x * this.x) + (this.y * this.y));
	}
	/**
	 * Adds two given Vector2f instances and stores the value in a third instance. 
	 * @param pv1 The first instance.
	 * @param pv2 The second instance.
	 * @param pvret The result instance.
	 */
    public static void add(Vector2d pv1, Vector2d pv2, Vector2d pvret) {
        pvret.x = pv1.x + pv2.x;
        pvret.y = pv1.y + pv2.y;
    }

    /**
     * Adds two given Vector2f instances and returns a new result instance. 
     * @param pv1 The first instance.
     * @param pv2 The second instance.
     * @return Resulting vector of pv1 + pv2.
     */
    public static Vector2d add(Vector2d pv1, Vector2d pv2) {
        return new Vector2d(pv1.x + pv2.x, pv1.y + pv2.y);
    }
    
    /**
     * Subtracts two given Vector2f instances and stored the result
     * in a third Vector2f instance. 
     * @param pv1 The first instance.
     * @param pv2 The second instance.
     * @param pvret The Third instance (result).
     */
    public static void sub(Vector2d pv1, Vector2d pv2, Vector2d pvret) {
        pvret.x = pv1.x - pv2.x;
        pvret.y = pv1.y - pv2.y;
    }

    /**
     * Subtracts two given Vector2f instances and return a
     * new instance of Vector2f.
     * @param pv1 The first instance.
     * @param pv2 The second instance.
     * @return Resulting vector of pv1 - pv2.
     */
    public static Vector2d sub(Vector2d pv1, Vector2d pv2) {
        return new Vector2d(pv1.x - pv2.x, pv1.y - pv2.y);
    }

    /**
     * Multiplies a Vector2f instance with a given scalar value and stores
     * the result in a second Vector2f instance.
     * @param pv An instance of Vector2f.
     * @param pscalar A scalar value.
     * @param presult The second instance (result).
     */
    public static void mul(Vector2d pv, double pscalar, Vector2d presult) {
		presult.x = pv.x * pscalar;
		presult.y = pv.y * pscalar;
	}
    /**
     * Returns the scalar product of two vectors.
     * @param pv1 Left operand vector.
     * @param pv2 Right operand vector.
     * @return The scalar product.
     */
    public static double scalar(Vector2d pv1, Vector2d pv2) {
        return (pv1.x * pv2.x) + (pv1.y * pv2.y);
    }

    /**
     * Returns the cosinus of the angle between two vectors.
     * @param pv1 The first vector.
     * @param pv2 The second vector.
     * @return The angle.
     */
    public static double cos(Vector2d pv1, Vector2d pv2) {
        final double scalar = scalar(pv1, pv2);
        final double length = (pv1.length() * pv2.length());
        return scalar / length;
    }
    
    /**
     * Returns the angle (phi) between two vectors.
     * @param pv1 The first vector.
     * @param pv2 The second vector.
     * @return The angle.
     */
    public static double phi(Vector2d pv1, Vector2d pv2) {
        return (double)Math.acos(cos(pv1, pv2));
    }
    
    /**
     * Returns the angle (phi) between two vectors signed (counterclockwise).
     * @param pv1 The first vector.
     * @param pv2 The second vector.
     * @return The angle.
     */
    public static double signedPhi(Vector2d pv1, Vector2d pv2) {
        //
        final double a1 = Math.atan2(pv2.y, pv2.x);
        final double a2 = Math.atan2(pv1.y, pv1.x);
        //
        return (double)(a1 - a2);
    }
    /**
     * Multiplies a given vector with a scalar value. The method changes
     * the given vector instance.
     * @param pv The vector.
     * @param pscalar The scalar value.
     * @return The scales vector.
     */
    public static Vector2d mul(Vector2d pv, double pscalar) {
		return new Vector2d(pv.x * pscalar, pv.y * pscalar);
	}
    /**
     * Rotates a vector and returns the result as new vector.
     * @param pv The vector which is to rotate.
     * @param pangle The angle in rad of the rotation.
     * @return The return vector.
     */    
    public static Vector2d rotate(Vector2d pv, double pangle) {
        Vector2d v = new Vector2d();
        rotate(pv, pangle, v);
        return v;
    }
    
    /**
     * Rotates a vector and stores the result into pvret.
     * @param pv The vector which is to rotate.
     * @param pangle The angle in rad of the rotation.
     * @param pvret The return vector.
     */    
    public static void rotate(Vector2d pv, double pangle, Vector2d pvret) {
        double cos = (double)Math.cos(pangle);
        double sin = (double)Math.sin(pangle);
        //
        final double x = pv.x;
        final double y = pv.y;
        //
        pvret.x = (x * cos) + (y * -sin);  
        pvret.y = (x * sin) + (y * cos);         
    }
			
}
