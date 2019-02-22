package de.cogmod.utilities;

public class ActivationFunctions {

	public static enum ACT_FUNCT{
		Linear,
		Sigmoid,
		Tanh
	};

	
	
	public static double getLowerBound(ACT_FUNCT af) {
		switch(af) {
		case Linear:
			return -1e10;
		case Sigmoid:
			return 0;
		case Tanh:
			return -1;
		}
		System.err.println("Do not know activation function: "+af);
		return Double.NaN;		
	}
	
	public static double getUpperBound(ACT_FUNCT af) {
		switch(af) {
		case Linear:
			return 1e10;
		case Sigmoid:
			return 1;
		case Tanh:
			return 1;
		}
		System.err.println("Do not know activation function: "+af);
		return Double.NaN;		
	}

	public static double getFofX(final double x, ACT_FUNCT af) {
		switch(af) {
		case Linear:
			return x;
		case Sigmoid:
			return sigmoid(x);
		case Tanh:
			return tanh(x);
		}
		System.err.println("Do not know activation function: "+af);
		return 0;
	}

	public static double getDerivativeFofX(final double x, ACT_FUNCT af) {
		switch(af) {
		case Linear:
			return 1;
		case Sigmoid:
			return ActivationFunctions.sigmoidDx(x);
		case Tanh:
			return ActivationFunctions.tanhDx(x);
		}
		System.err.println("Do not know activation function: "+af);
		return 0;
	}
	
    public static double sigmoid(final double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    public static double sigmoidDx(final double x) {
        final double sig = 1.0 / (1.0 + fastExp(-x));
        return sig * (1.0 - sig);
    }
    
    public static double sigmoidDxMath(final double x) {
        final double sig = 1.0 / (1.0 + Math.exp(-x));
        return sig * (1.0 - sig);
    }

	public static double tanh(final double x) {
		// (e^2x-1)/(e^2x+1) = 1-2/(e^2x+1)
		return 1.-2/(fastExp(2.*x)+1.);
	}

	public static double tanhMath(final double x) {
		return Math.tanh(x);
	}

	public static double tanhDx(final double x) {
		final double tanhx = 1.-2/(fastExp(2.*x)+1.);
		return 1.0 - (tanhx * tanhx);

	}

	public static double tanhDxMath(final double x) {
		final double tanhx = Math.tanh(x);
		return 1.0 - (tanhx * tanhx);

	}
	
	public static final int    fastExpN = 10;
    public static final double fastExpReciprocal = 1.0 / Math.pow(2, fastExpN); 
    
    public static double fastExp(final double x) {
        double v = 1.0 + (x * fastExpReciprocal);
        //
        for (int i = 0; i < fastExpN; i++) {
            v *= v;
        }
        //
        return v;
    }
    
    public static double fastFastExp(final double value) {
        final long tmp = (long)(1512775 * value) + 1072632447;
        return Double.longBitsToDouble(tmp << 32);
    }
    
	
}
