package hb.network;

import hb.layers.Layer;
import hb.layers.Loss;
import hb.matrix.Matrix;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Random;
import java.util.random.RandomGenerator;

public class Network {
    private Network() {}

    public static Matrix runNetwork(Layer[] network, Matrix input) {
        for (Layer layer : network) {
            input = layer.forwardMut(input);
        }

        return input;
    }

    public static GradientPair calculateGradients(Layer[] network, Loss loss, Matrix input, Matrix actual) {
        Matrix[] gradients = new Matrix[network.length];

        Deque<Matrix> stored = new ArrayDeque<>(network.length + 1);
        stored.push(input);

        for (Layer layer : network) {
            input = layer.forward(input);
            stored.push(input);
        }

        final float loss_v = loss.loss(stored.peek(), actual);
        Matrix inputGradient = loss.gradient(stored.peek(), actual);

        for (int i = network.length - 1; i >= 0; i--) {
            Matrix layerOutput = stored.pop();
            Matrix layerInput = stored.peek();
            gradients[i] = network[i].weightGradient(layerInput, layerOutput, inputGradient);
            // input gradient not needed for first layer
            if (i != 0) inputGradient = network[i].inputGradient(layerInput, layerOutput, inputGradient);
        }

        return new GradientPair(gradients, loss_v);
    }

    public static void randomizeWeights(Layer[] network, RandomGenerator random) {
        for (Layer layer : network)
            layer.initializeWeights(random);
    }

    public record GradientPair(Matrix[] gradients, float loss) {}
}
