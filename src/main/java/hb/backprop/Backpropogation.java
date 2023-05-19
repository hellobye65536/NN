package hb.backprop;

import hb.nn.Layer;
import hb.nn.Loss;
import hb.tensor.Matrix;

import java.util.Stack;

public class Backpropogation {
    Matrix[] calculateGradients(Layer[] layers, Loss loss, Matrix input, Matrix actual) {
        Matrix[] gradients = new Matrix[layers.length];

        Stack<Matrix> stored = new Stack<>();
        stored.push(input);

        Matrix cur = input;
        for (Layer layer : layers) {
            cur = layer.forward(cur);
            stored.push(cur);
        }

        System.out.println(stored);

        return gradients;
    }
}
