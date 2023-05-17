package hb.tensor;

import java.util.function.DoubleFunction;
import java.util.function.DoubleUnaryOperator;

public final class Matrix {
    private final float[] buffer;
    private final int rows, cols;

    public Matrix(float[] buffer, int rows, int cols) {
        if (buffer.length != rows * cols)
            throw new IllegalArgumentException();

        this.buffer = buffer;
        this.rows = rows;
        this.cols = cols;
    }

    public Matrix(float[][] data) {
        rows = data.length;

        if (rows == 0) {
            cols = 0;
            buffer = new float[0];
            return;
        }

        cols = data[0].length;
        for (int i = 1; i < data.length; i++) {
            if (data[i].length != cols)
                throw new IllegalArgumentException();
        }

        buffer = new float[rows * cols];
    }

    public static Matrix zeros(int rows, int cols) {
        return new Matrix(new float[rows * cols], rows, cols);
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public float get(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw new IndexOutOfBoundsException();

        return buffer[row * cols + col];
    }

    public void set(int row, int col, float v) {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw new IndexOutOfBoundsException();

        buffer[row * cols + col] = v;
    }

    public void mutate(DoubleUnaryOperator func) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = (float) func.applyAsDouble(buffer[i]);
        }
    }

    public Matrix matmulNN(Matrix lhs) {
        if (this.cols != lhs.rows)
            throw new IllegalArgumentException();

        Matrix result = Matrix.zeros(this.rows, lhs.cols);
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < lhs.cols; col++) {
                float element = 0;
                for (int i = 0; i < this.cols; i++) {
                    element += this.get(row, i) * lhs.get(i, col);
                }
                result.set(row, col, element);
            }
        }

        return result;
    }

    @Override
    protected Matrix clone() throws CloneNotSupportedException {
        return new Matrix(buffer.clone(), rows, cols);
    }
}
