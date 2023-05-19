package hb;

import hb.nn.*;
import hb.tensor.Matrix;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(
                        new GZIPInputStream(
                                new FileInputStream("misc/train.csv.gz")
                        )
                )
        );

        String[] split = reader.readLine().split(",");
        System.out.println(Arrays.toString(split));

        Layer[] network = new Layer[]{
                new Dense(28 * 28, 32),
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
    }
}