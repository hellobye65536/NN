package hb.layers;

import hb.matrix.Matrix;

public class CrossEntropy implements Loss {
    @Override
    public float loss(Matrix predicted, Matrix actual) {
        float total = 0;

        for (int row = 0; row < predicted.rows(); row++) {
            for (int col = 0; col < predicted.cols(); col++) {
                // have small number be minimum to avoid taking the log of zero
                final float v_p = (float) Math.max(predicted.get(row, col), 0.0001);
                final float v_a = actual.get(row, col);
                total += v_a * Math.log(v_p);
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
                // have small number be minimum to avoid dividing by zero
                final float v_p = (float) Math.max(predicted.get(row, col), 0.0001);
                final float v_a = actual.get(row, col);

                ret.set(row, col, -v_a / v_p / predicted.cols());
            }
        }

        return ret;
    }
}
