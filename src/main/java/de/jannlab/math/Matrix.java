package de.jannlab.math;

import java.io.Serializable;


/**
 * 
 * @author Sebastian Otte
 */
public class Matrix implements Serializable {
    private static final long serialVersionUID = 145796898686683862L;

    public final int rows;
    public final int cols;
    public final double[] data;

    public Matrix copy() {
        return new Matrix(this.rows, this.cols, this.data.clone());
    }
    
    public Matrix getRow(final int i) {
    	final Matrix result = new Matrix(1, this.cols); 
    	//
    	int offset = this.cols * i;
    	//
    	for (int j = 0; j < this.cols; j++) {
    		result.data[j] = this.data[offset];
    		offset++;
    	}
    	//
    	return result;
    }
    
    public Matrix getColumn(final int j) {
    	final Matrix result = new Matrix(this.rows, 1); 
    	//
    	int offset = j;
    	//
    	for (int i = 0; i < this.rows; i++) {
    		result.data[i] = this.data[offset];
    		offset += this.cols;
    	}
    	//
    	return result;
    }
    
    public Matrix selectRows(final int first, final int last) {
    	final int rows = (last - first) + 1;
    	final Matrix result = new Matrix(rows, this.cols);
    	//
    	int row = 0;
    	for (int i = first; i <= last; i++) {
    		for (int j = 0; j < this.cols; j++) {
    			result.set(row, j, this.get(i, j));
    		}
    		row++;
    	}
    	//
    	return result;
    }
    
    
    /**
     * TODO: move to Matrix or Matrix tools.
     */
    public Matrix selectColumns(final int ...columns) {
    	final Matrix result = new Matrix(this.rows, columns.length);
    	//
    	for (int i = 0; i < this.rows; i++) {
    		for (int j = 0; j < columns.length; j++) {
    			result.set(i, j, this.get(i, columns[j]));
    		}
    	}
    	//
    	return result;
    }    
    
    
    public double get(final int i, final int j) {
        return this.data[(i * this.cols) + j];
    }
    
    public void set(final int i, final int j, final double value) {
        this.data[(i * this.cols) + j] = value;
    }
    
    public Matrix transpose() {
    	final Matrix At = new Matrix(this.cols, this.rows);
    	MatrixTools.transpose(this.data, this.rows, this.cols, At.data);
    	return At;
    }
    
    
    @Override
    public String toString() {
    	return MatrixTools.asString(this, 4);
    }
    
    public Matrix(
        final int rows,
        final int cols,
        final double[] data
    ) {
        this.rows = rows;
        this.cols = cols;
        this.data = data;
    }
    
    public Matrix(
        final int rows,
        final int cols
    ) {
        this(
            rows, cols,
            new double[rows * cols]
        );
    }
}