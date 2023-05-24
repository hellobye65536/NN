package hb.tensor;

import java.util.Arrays;

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

    public Matrix transpose() {
        Matrix transposed = Matrix.zeros(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.set(j, i, get(i, j));
            }
        }

        return transposed;
    }

    public Matrix mulScalar(float v) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] *= v;
        }

        return this;
    }

    public Matrix add(Matrix rhs) {
        if (this.rows != rhs.rows || this.cols != rhs.cols)
            throw new IllegalArgumentException();

        for (int i = 0; i < buffer.length; i++) {
            this.buffer[i] += rhs.buffer[i];
        }

        return this;
    }

    public Matrix matmulNN(Matrix rhs) {
        if (this.cols != rhs.rows)
            throw new IllegalArgumentException();

        Matrix result = Matrix.zeros(this.rows, rhs.cols);
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < rhs.cols; col++) {
                float element = 0;
                for (int i = 0; i < this.cols; i++) {
                    element += this.get(row, i) * rhs.get(i, col);
                }
                result.set(row, col, element);
            }
        }

        return result;
    }

    public Matrix matmulTN(Matrix rhs) {
        if (this.rows != rhs.rows)
            throw new IllegalArgumentException();

        Matrix result = Matrix.zeros(this.cols, rhs.cols);
        for (int row = 0; row < this.cols; row++) {
            for (int col = 0; col < rhs.cols; col++) {
                float element = 0;
                for (int i = 0; i < this.rows; i++) {
                    element += this.get(i, row) * rhs.get(i, col);
                }
                result.set(row, col, element);
            }
        }

        return result;
    }

    public Matrix matmulNT(Matrix rhs) {
        if (this.cols != rhs.cols)
            throw new IllegalArgumentException();

        Matrix result = Matrix.zeros(this.rows, rhs.rows);
        for (int row = 0; row < this.rows; row++) {
            for (int col = 0; col < rhs.rows; col++) {
                float element = 0;
                for (int i = 0; i < this.cols; i++) {
                    element += this.get(row, i) * rhs.get(col, i);
                }
                result.set(row, col, element);
            }
        }

        return result;
    }

    public void throwIfNaN() throws ArithmeticException {
        for (int i = 0; i < buffer.length; i++) {
            if (Float.isNaN(buffer[i]))
                throw new ArithmeticException();
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Matrix matrix = (Matrix) o;

        if (rows != matrix.rows) return false;
        if (cols != matrix.cols) return false;
        return Arrays.equals(buffer, matrix.buffer);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(buffer);
        result = 31 * result + rows;
        result = 31 * result + cols;
        return result;
    }

    @Override
    public Matrix clone() {
        return new Matrix(buffer.clone(), rows, cols);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");

        for (int row = 0; row < rows; row++) {
            sb.append("\t[");
            for (int col = 0; col < cols; col++) {
                sb.append(get(row, col));
                sb.append(", ");
            }
            sb.append("],\n");
        }

        sb.append("]");

        return sb.toString();
    }
}
