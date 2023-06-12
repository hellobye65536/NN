package hb.app;

import hb.layers.*;
import hb.matrix.Matrix;

import java.io.*;
import java.util.Arrays;

public class Model {
    /**
     * The width of the image
     */
    public static final int IMAGE_WIDTH = 28;
    /**
     * The pixel count of the image
     */
    public static final int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_WIDTH;

    /**
     * The loss function used in training
     */
    public static final Loss loss = new CrossEntropy();

    /**
     * @return The network used to predict digits
     */
    public static Layer[] buildNetwork() {
        return new Layer[] {
            new Dense(IMAGE_SIZE, 64),
            new Bias(64),
            new ReLU(),
            new Dense(64, 32),
            new Bias(32),
            new ReLU(),
            new Dense(32, 10),
            new Bias(10),
            new Softmax(),
            };
    }

    /**
     * Saves weights from <code>network</code> into the InputStream <code>stream</code>
     *
     * @param network The network
     * @param stream  The input stream
     * @throws IOException If there was an error
     */
    public static void saveWeights(Layer[] network, OutputStream stream) throws IOException {
        final DataOutputStream weightOut = new DataOutputStream(stream);

        for (Layer layer : network) {
            if (layer.weights() == null)
                continue;

            final Matrix weights = layer.weights();
            for (int row = 0; row < weights.rows(); row++) {
                for (int col = 0; col < weights.cols(); col++) {
                    weightOut.writeFloat(weights.get(row, col));
                }
            }
        }

        weightOut.flush();
    }

    /**
     * Loads weights into <code>network</code> from the InputStream <code>stream</code>
     *
     * @param network The network
     * @param stream  The input stream
     * @throws IOException If there was an error
     */
    public static void loadWeights(Layer[] network, InputStream stream) throws IOException {
        DataInputStream weightIn = new DataInputStream(stream);
        for (Layer layer : network) {
            final Matrix weights = layer.weights();
            if (weights == null)
                continue;

            for (int row = 0; row < weights.rows(); row++) {
                for (int col = 0; col < weights.cols(); col++) {
                    weights.set(row, col, weightIn.readFloat());
                }
            }
        }
    }


    /**
     * Represents a raw pair of data: correct labels, and image data
     *
     * @param labels The correct labels
     * @param data The image data flattened
     */
    public record DataPair(int[] labels, float[] data) {
        /**
         * Load a <code>DataPair</code> from the reader
         * @param reader The BufferedReader
         * @return a parsed DataPair
         * @throws IOException if there was an exception
         */
        public static DataPair loadData(BufferedReader reader) throws IOException {
            int length = Integer.parseInt(reader.readLine());

            int[] labels = new int[length];
            float[] data = new float[length * IMAGE_SIZE];

            for (int i = 0; i < length; i++) {
                int[] values = Arrays.stream(reader.readLine().split(",")).mapToInt(Integer::parseInt).toArray();

                if (values.length != IMAGE_SIZE + 1) throw new RuntimeException("Data not in correct format");

                labels[i] = values[0];
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    data[j + i * IMAGE_SIZE] = values[1 + j] / 255.0f;
                }
            }

            return new DataPair(labels, data);
        }

        /**
         * Process a subset of this data into matrices, in the form of a <code>{@link ProcessedPair}</code>
         *
         * @param indices An array of indices of the data
         * @param begin The beginning (inclusive) of the range of indices to process
         * @param end The end (exclusive) of the range of indices to process
         * @return A <code>{@link ProcessedPair}</code>
         */
        public ProcessedPair processData(int[] indices, int begin, int end) {
            if (begin >= end) throw new IllegalArgumentException();

            Matrix input = Matrix.zeros(IMAGE_SIZE, end - begin);
            Matrix actual = Matrix.zeros(10, end - begin);

            for (int col = 0; col < end - begin; col++) {
                for (int row = 0; row < IMAGE_SIZE; row++) {
                    input.set(row, col, this.data[IMAGE_SIZE * indices[col + begin] + row]);
                }

                actual.set(this.labels[indices[col + begin]], col, 1);
            }

            return new ProcessedPair(input, actual);
        }

        public int size() {
            return labels.length;
        }
    }

    /**
     * Some subset of data processed into matrices, to be used by a neural network
     *
     * @param input The image data as matrices
     * @param actual The labels as matrices
     */
    public record ProcessedPair(Matrix input, Matrix actual) {}
}
