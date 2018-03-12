# import the necessary packages
import imutils
from sklearn.feature_extraction import image as imageLib
import numpy as np
import argparse
import time
import os
import cv2
import cntk as C
from helper import pyramid, sliding_window, slideWithPyramid, slide
import MyDeserializer

class LogisticRegression():
    def __init__(self, directory, input_dim, num_output_classes):
        self._directory = directory
        self._input_dim = input_dim
        self._num_output_classes = num_output_classes

# len = len(os.listdir(directory))


# deseralizer = MyDeserializer.MyDeserializer(directory=dataDirectory, streams=[dict(name='Features', shape=(361,)), dict(name='Labels', shape=(2,))])
#
# deseralizer.createDictionary(dataDirectory)
#
# deseralizer.get_chunk(2)


# Define the data dimensions



# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
    def create_reader(self, path, is_training, input_dim, num_label_classes):
        labelStream = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
        featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)

        # deserializer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels=labelStream, features=featureStream))
        deserializer = MyDeserializer.MyDeserializer(directory=path, streams=[dict(name='features', shape=(input_dim,)),
                                                                              dict(name='labels',
                                                                                   shape=(self._num_output_classes,))])

        return C.io.MinibatchSource(deserializer,
                                    randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)


    def create_model(self, features):
        with C.layers.default_options(init=C.glorot_uniform()):
            r = C.layers.Dense(self._num_output_classes, activation=None)(features)
            return r

        # Define a utility function to compute the moving average sum.
        # A more efficient implementation is possible with np.cumsum() function


    def moving_average(self, a, w=5):
        if len(a) < w:
            return a[:]  # Need to send a copy of the array
        return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


    # Defines a utility that prints the training progress
    def print_training_progress(self, trainer, mb, frequency, verbose=1):
        training_loss = "NA"
        eval_error = "NA"

        if mb % frequency == 0:
            training_loss = trainer.previous_minibatch_loss_average
            eval_error = trainer.previous_minibatch_evaluation_average
            if verbose:
                print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error * 100))

        return mb, training_loss, eval_error


    def testSolution(self, test_file, trainer):
        input = C.input_variable(self._input_dim)
        label = C.input_variable(self._num_output_classes)
        # Read the training data
        reader_test = self.create_reader(test_file, False, self._input_dim, self._num_output_classes)

        test_input_map = {
            label: reader_test.streams.labels,
            input: reader_test.streams.features,
        }

        # Test data for trained model
        test_minibatch_size = 512
        num_samples = 10000
        num_minibatches_to_test = num_samples // test_minibatch_size
        test_result = 0.0

        for i in range(num_minibatches_to_test):
            # We are loading test data in batches specified by test_minibatch_size
            # Each data point in the minibatch is a MNIST digit image of 784 dimensions
            # with one pixel per dimension that we will encode / decode with the
            # trained model.
            data = reader_test.next_minibatch(test_minibatch_size,
                                              input_map=test_input_map)

            eval_error = trainer.test_minibatch(data)
            test_result = test_result + eval_error

        # Average of evaluation errors of all test minibatches
        print("Average test error: {0:.2f}%".format(test_result * 100 / num_minibatches_to_test))


    def start(self):
        input = C.input_variable(self._input_dim)
        label = C.input_variable(self._num_output_classes)

        # Scale the input to 0-1 range by dividing each pixel by 255.
        z = self.create_model(input / 255.0)
        loss = C.cross_entropy_with_softmax(z, label)
        label_error = C.classification_error(z, label)

        # Instantiate the trainer object to drive the model training
        learning_rate = 0.2
        lr_schedule = C.learning_parameter_schedule(learning_rate)
        learner = C.sgd(z.parameters, lr_schedule)
        trainer = C.Trainer(z, (loss, label_error), [learner])

        # Initialize the parameters for the trainer
        minibatch_size = 64
        num_samples_per_sweep = 60000
        num_sweeps_to_train_with = 10
        num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

        # Create the reader to training data set
        # TODO: check if train_file replaced by dataDirectory works
        reader_train = self.create_reader(self._directory, True, self._input_dim, self._num_output_classes)

        # Map the data streams to the input and labels.
        input_map = {
            label: reader_train.streams.labels,
            input: reader_train.streams.features
        }

        # Run the trainer on and perform model training
        training_progress_output_freq = 500

        plotdata = {"batchsize": [], "loss": [], "error": []}

        for i in range(0, int(num_minibatches_to_train)):

            # Read a mini batch from the training data file
            data = reader_train.next_minibatch(minibatch_size, input_map=input_map)

            trainer.train_minibatch(data)
            batchsize, loss, error = self.print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

            if not (loss == "NA" or error == "NA"):
                plotdata["batchsize"].append(batchsize)
                plotdata["loss"].append(loss)
                plotdata["error"].append(error)

        # Compute the moving average loss to smooth out the noise in SGD
        # plotdata["avgloss"] = moving_average(plotdata["loss"])
        # plotdata["avgerror"] = moving_average(plotdata["error"])

        # Plot the training loss and the training error
        import matplotlib.pyplot as plt
        #
        # plt.figure(1)
        # plt.subplot(211)
        # plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
        # plt.xlabel('Minibatch number')
        # plt.ylabel('Loss')
        # plt.title('Minibatch run vs. Training loss')
        #
        # plt.show()
        #
        # plt.subplot(212)
        # plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
        # plt.xlabel('Minibatch number')
        # plt.ylabel('Label Prediction Error')
        # plt.title('Minibatch run vs. Label Prediction Error')
        # plt.show()

        self.testSolution(self._directory, trainer)

        print('end of program')


