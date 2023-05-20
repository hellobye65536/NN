package hb;

import hb.layers.*;
import hb.tensor.Matrix;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

public class Train {
    public static final int IMAGE_SIZE = 28 * 28;

    public static void main(String[] args) throws IOException {
        DataPair training = loadData(buildStream("./misc/train.csv.gz"));
        DataPair testing = loadData(buildStream("./misc/test.csv.gz"));

        Layer[] network = new Layer[] {
            new Dense(IMAGE_SIZE, 32),
            new ReLU(),
            new Dense(32, 32),
            new ReLU(),
            new Dense(32, 32),
            new ReLU(),
            new Dense(32, 32),
            new ReLU(),
            new Dense(32, 10),
            new Softmax(),
            };
        Loss loss = new CrossEntropy();


//        System.out.println(Network.runNetwork(network, t));

//        Backpropogation.calculateGradients(network, loss, )
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

    record DataPair(int[] labels, float[] data) {}

    record ProcessedPair(Matrix input, Matrix expectedOutput) {}
}