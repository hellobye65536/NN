package hb.network;

import hb.layers.Layer;
import hb.layers.Loss;
import hb.tensor.Matrix;

import java.util.ArrayDeque;
import java.util.Deque;

public class Network {
    private Network() {}

    public static Matrix runNetwork(Layer[] network, Matrix input) {
        for (Layer layer : network) {
            input = layer.forwardMut(input);
        }

        return input;
    }

    public static Matrix[] calculateGradients(Layer[] network, Loss loss, Matrix input, Matrix actual) {
        Matrix[] gradients = new Matrix[network.length];

        Deque<Matrix> stored = new ArrayDeque<>();
        stored.push(input);

        for (Layer layer : network) {
            input = layer.forward(input);
            stored.push(input);
        }

        System.out.println(stored);

        return gradients;
    }
}
