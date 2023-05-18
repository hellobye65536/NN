package hb.nn;

import hb.tensor.Matrix;

public class ReLU implements Layer {
    @Override
    public Matrix weights() {
        return null;
    }

    @Override
    public Matrix forward(Matrix input) {
        input = input.clone();
        for (int row = 0; row < input.rows(); row++) {
            for (int col = 0; col < input.cols(); col++) {
                final float v = input.get(row, col);
                input.set(row, col, Math.max(0, v));
            }
        }
        return input;
    }

    @Override
    public Matrix inputGradient(Matrix input, Matrix output, Matrix outputGradient) {
        for (int row = 0; row < input.rows(); row++) {
            for (int col = 0; col < input.cols(); col++) {
                if (input.get(row, col) < 0)
                    outputGradient.set(row, col, 0);
            }
        }

        return outputGradient;
    }

    @Override
    public Matrix weightGradient(Matrix input, Matrix output, Matrix outputGradient) {
        return null;
    }
}