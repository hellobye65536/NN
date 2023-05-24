package hb.network;

import hb.layers.Layer;
import hb.layers.Loss;
import hb.tensor.Matrix;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Random;

public class Network {
    private static Random random = new Random();

    private Network() {
    }

    public static Matrix runNetwork(Layer[] network, Matrix input) {
        for (Layer layer : network) {
            input = layer.forwardMut(input);
        }

        return input;
    }

    public static Matrix[] calculateGradients(Layer[] network, Loss loss, Matrix input, Matrix actual) {
        Matrix[] gradients = new Matrix[network.length];

        Deque<Matrix> stored = new ArrayDeque<>(network.length + 1);
        stored.push(input);

        for (Layer layer : network) {
            input = layer.forward(input);
            stored.push(input);
        }

        Matrix inputGradient = loss.gradient(stored.peek(), actual);

        for (int i = network.length - 1; i >= 0; i--) {
            Matrix layerOutput = stored.pop();
            Matrix layerInput = stored.peek();
            gradients[i] = network[i].weightGradient(layerInput, layerOutput, inputGradient);
            // input gradient not needed for first layer
            if (i != 0)
                inputGradient = network[i].inputGradient(layerInput, layerOutput, inputGradient);
        }

        return gradients;
    }

    public static void randomizeWeights(Layer[] network) {
        for (Layer layer : network)
            layer.initializeWeights(random);
    }
}
