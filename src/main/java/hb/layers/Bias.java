package hb.layers;

import hb.tensor.Matrix;

public class Bias implements Layer {
    private final Matrix weights;

    public Bias(int size) {
        weights = Matrix.zeros(size, 1);
    }

    @Override
    public Matrix weights() {
        return weights;
    }

    @Override
    public Matrix forwardMut(Matrix input) {
        for (int row = 0; row < input.rows(); row++) {
            for (int col = 0; col < input.cols(); col++) {
                final float v_i = input.get(row, col);
                final float v_w = weights.get(row, 0);
                input.set(row, col, v_i + v_w);
            }
        }

        return input;
    }

    @Override
    public Matrix inputGradient(Matrix input, Matrix output, Matrix outputGradient) {
        return outputGradient;
    }

    @Override
    public Matrix weightGradient(Matrix input, Matrix output, Matrix outputGradient) {
        Matrix grad = Matrix.zeros(weights.rows(), 1);

        for (int row = 0; row < grad.rows(); row++) {
            float sum = 0;
            for (int col = 0; col < outputGradient.cols(); col++) {
                sum += outputGradient.get(row, col);
            }
            grad.set(row, 0, sum);
        }

        return grad;
    }
}
