package hb.network;

import hb.layers.Layer;
import hb.layers.Loss;
import hb.matrix.Matrix;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.random.RandomGenerator;

/**
 * Contains utility methods related to networks (arrays of layers)
 */
public class Network {
    private Network() {}

    /**
     * Run the network on some input.
     *
     * @param network The network as an array of layers
     * @param input   The input matrix
     * @return The output of the network
     */
    public static Matrix runNetwork(Layer[] network, Matrix input) {
        boolean first = true;

        for (Layer layer : network) {
            // avoid mutating input
            if (first) {
                input = layer.forward(input);
            } else {
                input = layer.forwardMut(input);
            }

            first = false;
        }

        return input;
    }

    /**
     * Given a network, a loss function, an input, and the expected output, calculate the gradients of the weights with
     * respect to the loss.
     *
     * @param network The network as an array of layers
     * @param loss    The loss function
     * @param input   The input matrix
     * @param actual  The expected output matrix
     * @return The gradients, and the loss for this input and output pair
     */
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

    /**
     * Randomize the weights in the network according to the layer type
     *
     * @param network The network
     * @param random  The source of randomness
     */
    public static void randomizeWeights(Layer[] network, RandomGenerator random) {
        for (Layer layer : network)
            layer.initializeWeights(random);
    }

    /**
     * Returned by <code>calculateGradients</code>. Contains gradients and a loss.
     *
     * @param gradients The gradients
     * @param loss      The loss
     */
    public record GradientPair(Matrix[] gradients, float loss) {}
}
