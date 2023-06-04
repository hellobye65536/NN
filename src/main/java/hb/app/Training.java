package hb.app;

import hb.app.Model.DataPair;
import hb.app.Model.ProcessedPair;
import hb.layers.Layer;
import hb.layers.Loss;
import hb.matrix.Matrix;
import hb.network.Network;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;

import java.io.*;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

import static hb.app.Model.saveWeights;

public class Training {

    public static final int EPOCHS = 100;
    public static final int BATCH_SIZE = 256;
    public static final float TRAINING_RATE = 0.05f;

    public static void main(String[] args) {
        DataPair training, testing;
        try {
            training = DataPair.loadData(buildStream("./dataset/train.csv.gz"));
            testing = DataPair.loadData(buildStream("./dataset/test.csv.gz"));
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
                for (int batch = 0; batch * BATCH_SIZE < training.size(); batch++) {
                    final int batch_begin = batch * BATCH_SIZE;
                    final int batch_end = Math.min(batch_begin + BATCH_SIZE, training.size());

                    final ProcessedPair trainingBatch = training.processData(trainingIndices, batch_begin, batch_end);
                    final Network.GradientPair gradient_pair = Network.calculateGradients(network, loss,
                        trainingBatch.input(), trainingBatch.actual());

                    running_loss += gradient_pair.loss();
                    final Matrix[] gradients = gradient_pair.gradients();

                    for (int i = 0; i < network.length; i++) {
                        if (network[i].weights() != null) {
                            gradients[i].mulScalar(-TRAINING_RATE);
                            network[i].weights().add(gradients[i]);
                        }
                    }

                    bar.stepBy(batch_end - batch_begin);
                    bar.setExtraMessage("Loss: " + (running_loss / (batch + 1)));
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
                for (int batch = 0; batch * BATCH_SIZE < testing.size(); batch++) {
                    final int batch_begin = batch * BATCH_SIZE;
                    final int batch_end = Math.min(batch_begin + BATCH_SIZE, testing.size());

                    final ProcessedPair testingBatch = testing.processData(testingIndices, batch_begin, batch_end);
                    Matrix predicted = Network.runNetwork(network, testingBatch.input());
                    running_loss += loss.loss(predicted, testingBatch.actual());

                    bar.stepBy(batch_end - batch_begin);
                    bar.setExtraMessage("Loss: " + (running_loss / (batch + 1)));
                }
            }
        }

        try (OutputStream weightOut = new BufferedOutputStream(new FileOutputStream("trained_weights"))) {
            saveWeights(network, weightOut);
        } catch (IOException e) {
            System.err.println("Error writing trained weights");
            e.printStackTrace();
        }
    }

    private static BufferedReader buildStream(String path) throws IOException {
        return new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(path))));
    }

    /**
     * Shuffles the given array of integers. Uses a Fisher-Yates shuffle.
     */
    private static void shuffle(int[] arr) {
        Random random = new Random();
        for (int i = arr.length - 1; i > 0; i--) {
            final int j = random.nextInt(i + 1);

            final int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
}