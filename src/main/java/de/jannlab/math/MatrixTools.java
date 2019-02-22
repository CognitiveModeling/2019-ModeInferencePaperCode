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



import de.jannlab.tools.DoubleTools;
import static de.jannlab.tools.StringTools.*;

/**
 * @author Sebastian Otte
 *
 */
public class MatrixTools {
    
    public static final int VALUE_PADDING = 2;
    
    public static double[] allocate(final int m, final int n) {
        return new double[m * n];
    }
    
    public static double[] allocate(final int m) {
        return new double[m];
    }
    
    public static void setRow(
        final double[] matrix, final int n, final int m, 
        final int i, final double ...values
    ) {
        //
        setRow(matrix, n, m, i, values, 0);
    }
    
    public static void setRow(
        final double[] matrix, final int n, final int m, 
        final int i, final double[] values, final int offset
    ) {
        //
        DoubleTools.copy(
            matrix, i * n, values, offset, 
            Math.min(n, values.length - offset)
        );
    }
    
    public static int idx(final int i, final int j, final int n) {
        return (i * n) + j;
    }
    
    
    public static void add(
        final Matrix A, 
        final Matrix B, 
        final Matrix R
    ) {
        //
        add(A.data, B.data, R.data);
        //
    }

    public static void add(
        final double A[], 
        final double B[], 
        final double R[]
    ) {
        if (
            (A.length != B.length) ||
            (B.length != R.length)
        ) throw new RuntimeException("matrices do not match.");
        //
        DoubleTools.add(A, 0, B, 0, R, 0, R.length);
    }

    public static void sub(
        final Matrix A, 
        final Matrix B, 
        final Matrix R
    ) {
        //
        sub(A.data, B.data, R.data);
        //
    }
    
    public static void sub(
        final double A[], 
        final double B[], 
        final double R[]
    ) {
        if (
            (A.length != B.length) ||
            (B.length != R.length)
        ) throw new RuntimeException("matrices do not match.");
        //
        DoubleTools.sub(A, 0, B, 0, R, 0, R.length);
    }
    
    public static Matrix appendRight(final Matrix ...input) {
        int cols_sum = 0;
        int rows_max = 0;
        //
        for (int m = 0; m < input.length; m++) {
            cols_sum += input[m].cols;
            rows_max = Math.max(rows_max, input[m].rows);
        }
        //
        Matrix target = new Matrix(rows_max, cols_sum);
        appendRightRef(target, input);
        return target;
    }
        
    public static void appendRightRef(final Matrix target, final Matrix ...input) {
        //
        int cols_sum = 0;
        int rows_max = 0;
        //
        for (int m = 0; m < input.length; m++) {
            cols_sum += input[m].cols;
            rows_max = Math.max(rows_max, input[m].rows);
        }
        //
        assert(target.rows == rows_max);
        assert(target.cols == cols_sum);
        //
        int c = 0;
        for (int m = 0; m < input.length; m++) {
            Matrix A = input[m];
            for (int j = 0; j < A.cols; j++) {
                for (int i = 0; i < A.rows; i++) {
                    target.set(i, c, A.get(i, j));
                }
                c++;
            }
        }
    }
    
    public static Matrix appendVertical(final Matrix ...input) {
        int rows_sum = 0;
        int cols_max = 0;
        //
        for (int m = 0; m < input.length; m++) {
            rows_sum += input[m].rows;
            cols_max = Math.max(cols_max, input[m].cols);
        }
        //
        Matrix target = new Matrix(rows_sum, cols_max);
        appendVerticalRef(target, input);
        return target;
    }    
    
    public static void appendVerticalRef(final Matrix target, final Matrix ...input) {
        //
        int rows_sum = 0;
        int cols_max = 0;
        //
        for (int m = 0; m < input.length; m++) {
            rows_sum += input[m].rows;
            cols_max = Math.max(cols_max, input[m].cols);
        }
        //
        assert(target.cols == cols_max);
        assert(target.rows == rows_sum);
        //
        int r = 0;
        for (int m = 0; m < input.length; m++) {
            Matrix A = input[m];
            for (int i = 0; i < A.rows; i++) {
                for (int j = 0; j < A.cols; j++) {
                    target.set(r, j, A.get(i, j));
                }
            	r++;
            }
        }
    }
    
    
    public static void mul(final Matrix A, final Matrix B, final Matrix R) {
        //
        mul(
            A.data, A.rows, A.cols,
            B.data, B.rows, B.cols,
            R.data
        );
        //
    }

    public static void mulSave(final Matrix A, final Matrix B, final Matrix R) {
        //
        mul(
            A.data.clone(), A.rows, A.cols,
            B.data.clone(), B.rows, B.cols,
            R.data
        );
        //
    }
    
    
    public static void identify(final Matrix A) {
    	final int min = Math.min(A.cols, A.rows);
    	DoubleTools.fill(A.data, 0, A.data.length, 0.0);
    	//
    	for (int i = 0; i < min; i++) {
    		A.set(i, i, 1.0);
    	}
    }
    
    
    public static void mul(
        final double A[], final int ma, final int na,
        final double B[], final int mb, final int nb,
        final double R[]
    ) {
        if (na != mb) throw new RuntimeException("matrices do not match.");
        //
        final int mr   = ma;
        final int nr   = nb;
        //
        int offr     = 0;
        int row_offa = 0;
        //
        for (int i = 0; i < mr; i++) {
            for (int j = 0; j < nr; j++) {
                //
                R[offr]  = 0.0;
                int offa = row_offa;
                int offb = j;
                //
                for (int k = 0; k < na; k++) {
                    R[offr] += A[offa] * B[offb];
                    offa += 1;
                    offb += nb;
                }
                //
                offr++;
            }
            row_offa += na;
        }
    }
    
    public static void transpose(
        final Matrix A,
        final Matrix At
    ) {
        //
        transpose(A.data, A.rows, A.cols, At.data);
        //
    }
    
    public static void transpose(
        final double[] A,
        final int m,
        final int n,
        final double[] At
    ) {
        //
        final double[] copy = A.clone();
        final int      size = m * n;
        //
        int offset = 0;
        //
        for (int i = 0; i < size; i++) {
            //
            At[offset] = copy[i];
            //
            offset += m;
            if (offset >= size) {
                offset = (offset - size) + 1;
            }
        }
    }
    
    public static String asString(final Matrix A, final int decimals) {
    	return asString(A.data, A.rows, A.cols, decimals);
    }
    

    
    public static String asString(
        final double[] A,
        final int m,
        final int n,
        final int decimals
    ) {
        final StringBuilder out = new StringBuilder();
        final int      size     = m * n;
        final String[] elements = new String[size];
        //
        int max = 0;
        //
        for (int i = 0; i < size; i++) {
            elements[i] = DoubleTools.asString(A, i, 1, decimals);
            if (elements[i].length() > max) {
                max = elements[i].length();
            }
        }
        //
        final int celllength = max + VALUE_PADDING;
        //
        for (int i = 0; i < size; i++) {
            if ((i > 0) && ((i % n) == 0)) out.append(LINEBREAK);
            //
            final String value = elements[i];
            final int diff = celllength - value.length();
            for (int j = 0; j < diff; j++) {
                out.append(" ");
            }
            out.append(value);
        }
        //
        return out.toString();
    }
    
    
    
}
