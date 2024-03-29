package hb.layers;

import hb.matrix.Matrix;

import java.util.Random;
import java.util.random.RandomGenerator;

/**
 * Represents a dense layer, or a matrix multiplication by weights.
 */
public class Dense implements Layer {
    private final Matrix weights;

    public Dense(int input, int output) {
        weights = Matrix.zeros(output, input);
    }

    @Override
    public Matrix weights() {
        return weights;
    }

    @Override
    public Matrix forward(Matrix input) {
        return forwardMut(input);
    }

    @Override
    public Matrix forwardMut(Matrix input) {
        return weights.matmulNN(input);
    }

    @Override
    public Matrix inputGradient(Matrix input, Matrix output, Matrix outputGradient) {
        return weights.matmulTN(outputGradient);
    }

    @Override
    public Matrix weightGradient(Matrix input, Matrix output, Matrix outputGradient) {
        return outputGradient.matmulNT(input);
    }

    // using "He" initialization
    @Override
    public void initializeWeights(RandomGenerator random) {
        final double deviation = Math.sqrt(2.0 / weights.rows());

        for (int row = 0; row < weights.rows(); row++) {
            for (int col = 0; col < weights.cols(); col++) {
                weights.set(row, col, (float) (random.nextGaussian() * deviation));
            }
        }
    }
}
