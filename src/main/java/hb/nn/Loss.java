package hb.nn;

import hb.tensor.Matrix;

public interface Loss {
    float loss(Matrix predicted, Matrix actual);

    Matrix gradient(Matrix predicted, Matrix actual);
}
