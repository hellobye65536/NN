package hb.matrix;

import java.util.Arrays;

/**
 * Represents some matrix of floating point numbers.
 */
public final class Matrix {
    // The actual buffer storing the contents of this matrix
    private final float[] buffer;
    // The row and column counts of this matrix
    private final int rows, cols;

    /**
     * Construct a matrix using a provided buffer with some shape.
     *
     * @param buffer The provided buffer
     * @param rows   Rows
     * @param cols   Columns
     * @throws IllegalArgumentException if the length of <code>buffer</code> doesn't equal to <code>rows * cols</code>
     */
    public Matrix(float[] buffer, int rows, int cols) {
        if (buffer.length != rows * cols)
            throw new IllegalArgumentException();

        this.buffer = buffer;
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Construct a matrix filled with zeroes with some shape.
     *
     * @param rows Rows
     * @param cols Columns
     * @return The matrix
     */
    public static Matrix zeros(int rows, int cols) {
        return new Matrix(new float[rows * cols], rows, cols);
    }

    /**
     * @return The amount of rows in this matrix
     */
    public int rows() {
        return rows;
    }

    /**
     * @return The amount of columns in this matrix
     */
    public int cols() {
        return cols;
    }

    /**
     * Retrieves the value at the specified row and column in the matrix.
     *
     * @param row The row index (0-based)
     * @param col The column index (0-based)
     * @return The value at the specified position in the matrix
     * @throws IndexOutOfBoundsException if the row or column index is out of range
     */
    public float get(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw new IndexOutOfBoundsException();

        return buffer[row * cols + col];
    }

    /**
     * Modifies the value at the specified row and column in the matrix.
     *
     * @param row The row index (0-based)
     * @param col The column index (0-based)
     * @param v   The new value to set at the specified position in the matrix
     * @throws IndexOutOfBoundsException if the row or column index is out of range
     */
    public void set(int row, int col, float v) {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw new IndexOutOfBoundsException();

        buffer[row * cols + col] = v;
    }

    /**
     * Multiplies each element of this matrix by the specified scalar value.
     *
     * @param v The scalar value to multiply with
     * @return The updated matrix with the elements multiplied by the scalar value
     */
    public Matrix mulScalar(float v) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] *= v;
        }

        return this;
    }

    /**
     * Add the elements of <code>rhs</code> to this matrix element-wise.
     *
     * @param rhs The matrix to add to this one
     * @return This matrix updated
     */
    public Matrix add(Matrix rhs) {
        if (this.rows != rhs.rows || this.cols != rhs.cols)
            throw new IllegalArgumentException();

        for (int i = 0; i < buffer.length; i++) {
            this.buffer[i] += rhs.buffer[i];
        }

        return this;
    }

    /**
     * Calculate the matrix multiplication of this matrix and the matrix <code>rhs</code>
     *
     * @param rhs The other matrix
     * @return The result matrix
     */
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

    /**
     * Calculate the matrix multiplication of the transpose of this matrix and the matrix <code>rhs</code>
     *
     * @param rhs The other matrix
     * @return The result matrix
     */
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

    /**
     * Calculate the matrix multiplication of this matrix and the transpose of the matrix <code>rhs</code>
     *
     * @param rhs The other matrix
     * @return The result matrix
     */
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

    /**
     * Creates a reference to this matrix with a different shape but same buffer. Matrices are stored in a row-major
     * way.
     *
     * @param newRows The number of rows in the new matrix
     * @param newCols The number of columns in the new matrix
     * @return The reshaped matrix
     */
    public Matrix reshape(int newRows, int newCols) {
        return new Matrix(buffer, newRows, newCols);
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
