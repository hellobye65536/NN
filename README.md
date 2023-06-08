This is a basic neural network capable of recognizing digits from the MNIST dataset.

# Running
## Training
First, prepare a dataset. (The files `test.csv.gz` and `train.csv.gz`) Either use the provided data, or download MNIST using the python script. To run the python script, install the dependencies in `requirements.txt` and run it.

To train, run:
```bash
./gradlew runTraining
```

## Testing
To test an already trained network, run:
```bash
./gradlew runTesting
```

## Drawing
To test out the network graphically by drawing digits using a mouse, run: 
```bash
./gradlew runDrawUI
```

# Algorithms
## Backpropagation
The gradients used in training are calculated using backpropagation. 