package hb.app;

import hb.layers.Layer;
import hb.layers.Loss;
import hb.matrix.Matrix;
import hb.network.Network;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

import static hb.app.Model.IMAGE_SIZE;

public class Train {

    public static final int EPOCHS = 50;
    public static final int BATCH_SIZE = 64;

    public static void main(String[] args) {
        DataPair training, testing;
        try {
            training = loadData(buildStream("./dataset/train.csv.gz"));
            testing = loadData(buildStream("./dataset/test.csv.gz"));
        } catch (IOException e) {
            System.err.println("Error loading training/testing data");
            e.printStackTrace();
            return;
        }

        Layer[] network = Model.buildNetwork();
        Loss loss = Model.loss;

        Network.randomizeWeights(network, new Random());

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            System.out.printf("Epoch %d/%d:\n", epoch + 1, EPOCHS);

            int[] trainingIndices = IntStream.range(0, training.size()).toArray();
            shuffle(trainingIndices);

            // training for current epoch
            try (ProgressBar bar = new ProgressBarBuilder()
                .setTaskName("Training")
                .setInitialMax(training.size())
                .setUpdateIntervalMillis(500)
                .build()) {
                float running_loss = 0;
                for (int batch_begin = 0; batch_begin < training.size(); batch_begin += BATCH_SIZE) {
                    final int batch_end = Math.min(batch_begin + BATCH_SIZE, training.size());
                    final int cur_batch_size = batch_end - batch_begin;

                    final ProcessedPair trainingBatch = processData(training, trainingIndices, batch_begin, batch_end);
                    final Network.GradientPair gradient_pair = Network.calculateGradients(network, loss,
                        trainingBatch.input, trainingBatch.actual);

                    running_loss += gradient_pair.loss();
                    final Matrix[] gradients = gradient_pair.gradients();

                    for (int i = 0; i < network.length; i++) {
                        if (network[i].weights() != null) {
                            gradients[i].mulScalar(-0.001f);
                            network[i].weights().add(gradients[i]);
                        }
                    }

                    bar.stepBy(cur_batch_size);
                    bar.setExtraMessage(String.format("Loss: %f", running_loss / batch_end));
                }
            }

            int[] testingIndices = IntStream.range(0, testing.size()).toArray();

            // testing for current epoch
            try (ProgressBar bar = new ProgressBarBuilder()
                .setTaskName("Testing")
                .setInitialMax(testing.size())
                .setUpdateIntervalMillis(500)
                .build()) {
                float running_loss = 0;
                for (int batch_begin = 0; batch_begin < testing.size(); batch_begin += BATCH_SIZE) {
                    final int batch_end = Math.min(batch_begin + BATCH_SIZE, testing.size());
                    final int cur_batch_size = batch_end - batch_begin;

                    final ProcessedPair testingBatch = processData(testing, testingIndices, batch_begin, batch_end);
                    Matrix predicted = Network.runNetwork(network, testingBatch.input);
                    running_loss += loss.loss(predicted, testingBatch.actual);

                    bar.stepBy(cur_batch_size);
                    bar.setExtraMessage(String.format("Loss: %f", running_loss / batch_end));
                }
            }
        }

        try (DataOutputStream weightOut = new DataOutputStream(
            new BufferedOutputStream(new FileOutputStream("trained_weights")))) {
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
        } catch (IOException e) {
            System.err.println("Error writing trained weights");
            e.printStackTrace();
        }
    }

    private static ProcessedPair processData(DataPair data, int[] indices, int begin, int end) {
        if (begin >= end) throw new IllegalArgumentException();

        Matrix input = Matrix.zeros(IMAGE_SIZE, end - begin);
        Matrix actual = Matrix.zeros(10, end - begin);

        for (int col = 0; col < end - begin; col++) {
            for (int row = 0; row < IMAGE_SIZE; row++) {
                input.set(row, col, data.data[IMAGE_SIZE * indices[col + begin] + row]);
            }

            actual.set(data.labels[indices[col + begin]], col, 1);
        }

        return new ProcessedPair(input, actual);
    }

    private static DataPair loadData(BufferedReader reader) throws IOException {
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

    private static BufferedReader buildStream(String path) throws IOException {
        return new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(path))));
    }

    private static void shuffle(int[] arr) {
        Random random = new Random();
        for (int i = arr.length - 1; i > 0; i--) {
            final int j = random.nextInt(i + 1);

            final int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    record DataPair(int[] labels, float[] data) {
        public int size() {
            return labels.length;
        }
    }

    record ProcessedPair(Matrix input, Matrix actual) {}
}