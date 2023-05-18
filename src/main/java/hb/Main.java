package hb;

import hb.nn.Softmax;
import hb.tensor.Matrix;

public class Main {
    public static void main(String[] args) {
        final Matrix input = new Matrix(new float[] {1, 1, 1}, 3, 1);
        final Softmax softmax = new Softmax();
        final Matrix output = softmax.forward(input);

        System.out.println(input);
        System.out.println(output);
        System.out.println(softmax.inputGradient(input, output, new Matrix(new float[] {1, 1, 2}, 3, 1)));
    }
}