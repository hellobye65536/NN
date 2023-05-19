package hb.nn;

import hb.tensor.Matrix;

/**
 * Represents a single layer in a neural network.
 * <p>
 * It defines the methods required for forward propagation and backpropagation.
 */
public interface Layer {

    /**
     * @return the weights of the layer.
     */
    Matrix weights();

    /**
     * Performs forward propagation on the input matrix and returns the output matrix.
     * (Evaluates this layer with some input)
     *
     * @param input the input matrix to the layer.
     * @return the output matrix produced by the layer.
     */
    Matrix forward(Matrix input);

    /**
     * Calculates the gradient of the loss function with respect to the input of this layer,
     * given the gradient with respect to the output of this layer.
     *
     * @param input          the input matrix to the layer.
     * @param output         the output matrix from the layer.
     * @param outputGradient the gradient of the loss function with respect to the output of the layer.
     * @return the input gradient of the layer.
     */
    Matrix inputGradient(Matrix input, Matrix output, Matrix outputGradient);

    /**
     * Calculates the gradient of the loss function with respect to the weights of this layer,
     * given the gradient with respect to the output of this layer.
     *
     * @param input          the input matrix to the layer.
     * @param output         the output matrix from the layer.
     * @param outputGradient the gradient of the loss function with respect to the weights of the layer.
     * @return the weight gradient of the layer.
     */
    Matrix weightGradient(Matrix input, Matrix output, Matrix outputGradient);
}
