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
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public class Testing {

    public static final int BATCH_SIZE = 256;

    public static void main(String[] args) {
        DataPair testing;
        try {
            testing = DataPair.loadData(buildStream("./dataset/test.csv.gz"));
        } catch (IOException e) {
            System.err.println("Error loading training/testing data");
            e.printStackTrace();
            return;
        }

        Layer[] network = Model.buildNetwork();
        Loss loss = Model.loss;

        try (InputStream weightIn = new BufferedInputStream(new FileInputStream("trained_weights"))) {
            Model.loadWeights(network, weightIn);
        } catch (IOException e) {
            System.err.println("Error loading trained weights");
            e.printStackTrace();
            return;
        }

        int[] testingIndices = IntStream.range(0, testing.size()).toArray();

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

    private static BufferedReader buildStream(String path) throws IOException {
        return new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(path))));
    }
}