package hb.tensor.nn;

import hb.tensor.Matrix;

public class ReLU implements Layer {
    @Override
    public Matrix weights() {
        return null;
    }

    @Override
    public Matrix forward(Matrix input) {
        input.mutate(v -> Math.max(v, 0));
        return input;
    }

    @Override
    public Matrix inputGradient(Matrix input, Matrix outputGradient) {
        assert input.rows() == outputGradient.rows();
        assert input.cols() == outputGradient.cols();

        for (int row = 0; row < input.rows(); row++) {
            for (int col = 0; col < input.cols(); col++) {
                if (input.get(row, col) < 0)
                    outputGradient.set(row, col, 0);
            }
        }

        return outputGradient;
    }

    @Override
    public Matrix weightGradient(Matrix input, Matrix outputGradient) {
        return null;
    }
}
