This is a basic neural network capable of recognizing digits from the MNIST dataset.

# Running
This program requires at least JDK 17.
## Training
First, prepare a dataset. (The files `test.csv.gz` and `train.csv.gz`) Either use the provided data, or download MNIST using the python script. To run the python script, install the dependencies in `requirements.txt` and run it.

To train, run:
```bash
./gradlew runTraining
```
This will write a file called `trained_weights`.

## Testing
To test an already trained network, run:
```bash
./gradlew runTesting
```
This requires a weights file called `trained_weights`.

## Drawing
To test out the network graphically by drawing digits using a mouse, run: 
```bash
./gradlew runDrawUI
```
This requires a weights file called `trained_weights`.

The AI is sensitive to the position and size of the drawn digit. Try to approximately center the digit, and shift the digit around to achieve better results.
