package hb.app;

import hb.layers.*;

public class Model {
    public static final int IMAGE_SIZE = 28 * 28;

    public static Layer[] buildNetwork() {
        return new Layer[] {
            new Dense(IMAGE_SIZE, 32),
            new Bias(32),
            new ReLU(),
            new Dense(32, 32),
            new Bias(32),
            new ReLU(),
            new Dense(32, 10),
            new Bias(10),
            new Softmax(),
        };
    }
}
