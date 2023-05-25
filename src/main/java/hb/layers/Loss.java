package hb.layers;

import hb.matrix.Matrix;

public interface Loss {
    float loss(Matrix predicted, Matrix actual);

    Matrix gradient(Matrix predicted, Matrix actual);
}
