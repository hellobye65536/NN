package hb.nn;

import hb.tensor.Matrix;

public interface Loss {
    float loss(Matrix predicted, Matrix actual);

    float gradient(Matrix predicted, Matrix actual, float loss);
}
