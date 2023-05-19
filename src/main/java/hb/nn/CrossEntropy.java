package hb.nn;

import hb.tensor.Matrix;

public class CrossEntropy implements Loss {
    @Override
    public float loss(Matrix predicted, Matrix actual) {
        float total = 0;

        for (int row = 0; row < predicted.rows(); row++) {
            for (int col = 0; col < predicted.cols(); col++) {
                total += actual.get(row, col) * Math.log(predicted.get(row, col));
            }
        }

        total = -total / predicted.cols();

        return total;
    }

    @Override
    public Matrix gradient(Matrix predicted, Matrix actual) {
        Matrix ret = Matrix.zeros(predicted.rows(), predicted.cols());

        for (int row = 0; row < predicted.rows(); row++) {
            for (int col = 0; col < predicted.cols(); col++) {
                final float v_p = predicted.get(row, col);
                final float v_a = actual.get(row, col);
                ret.set(row, col, v_a / v_p);
            }
        }

        return ret;
    }
}
