package hb.layers;

import hb.matrix.Matrix;

import java.util.Random;
import java.util.random.RandomGenerator;

/**
 * Represents a softmax activation function.
 * <p>
 * See <a href="https://en.wikipedia.org/wiki/Softmax_function">Wikipedia</a>
 */
public class Softmax implements Layer {
    public final float beta;

    public Softmax() {
        this.beta = 1;
    }

    public Softmax(float beta) {
        this.beta = beta;
    }

    @Override
    public Matrix weights() {
        return null;
    }

    @Override
    public Matrix forwardMut(Matrix input) {
        for (int col = 0; col < input.cols(); col++) {
            float col_max = 0;
            for (int row = 0; row < input.rows(); row++) {
                col_max = Math.max(col_max, input.get(row, col));
            }

            for (int row = 0; row < input.rows(); row++) {
                final float v = input.get(row, col);
                input.set(row, col, (float) Math.exp(beta * (v - col_max)));
            }

            float col_sum = 0;
            for (int row = 0; row < input.rows(); row++) {
                col_sum += input.get(row, col);
            }

            for (int row = 0; row < input.rows(); row++) {
                final float v = input.get(row, col);
                input.set(row, col, v / col_sum);
            }
        }

        return input;
    }

    @Override
    public Matrix inputGradient(Matrix input, Matrix output, Matrix outputGradient) {
        for (int col = 0; col < input.cols(); col++) {
            float col_sum = 0;
            for (int row = 0; row < input.rows(); row++) {
                final float v_o = output.get(row, col);
                final float v_og = outputGradient.get(row, col);
                col_sum += v_o * v_og;
            }

            for (int row = 0; row < input.rows(); row++) {
                final float v_o = output.get(row, col);
                final float v_og = outputGradient.get(row, col);
                outputGradient.set(row, col, -beta * v_o * (col_sum - v_og));
            }
        }

        return outputGradient;
    }

    @Override
    public Matrix weightGradient(Matrix input, Matrix output, Matrix outputGradient) {
        return null;
    }

    @Override
    public void initializeWeights(RandomGenerator random) {}
}
