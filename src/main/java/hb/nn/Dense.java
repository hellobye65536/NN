package hb.nn;

import hb.tensor.Matrix;

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
}
