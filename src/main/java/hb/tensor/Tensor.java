package hb.tensor;

public interface Tensor {
    /**
     * Adds another tensor to this tensor element-wise.
     *
     * @param lhs The tensor to be added to this tensor
     * @return A new tensor representing the result of the addition
     * @throws UnsupportedOperationException If the operation is not supported
     */
    Tensor add(Tensor lhs);
}
