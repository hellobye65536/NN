package hb;

import hb.nn.Softmax;
import hb.tensor.Matrix;

public class Main {
    public static void main(String[] args) {
        Matrix input = new Matrix(new float[] {1, 3}, 2, 1);
        System.out.println(input);

        System.out.println(new Softmax((float) Math.log(2)).forward(input));
    }
}