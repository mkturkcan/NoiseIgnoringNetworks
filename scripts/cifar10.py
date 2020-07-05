import numpy as np
import autokeras as ak
from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# Prepare the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.ravel()
y_test = y_test.ravel()
# Prepare the noise dataset
x_noise = np.round(np.random.random(x_train.shape)*255.)
y_noise = np.random.randint(0, high=10, size=(x_train.shape[0],))
# Combine the two
x_all = np.vstack((x_train, x_noise))
y_all = np.hstack((y_train, y_noise))
# Normalize data
x_all = x_all / np.max(x_all) - 0.5
x_noise = x_noise / np.max(x_noise) - 0.5
x_train = x_train / np.max(x_train) - 0.5
x_test = x_test / np.max(x_test) - 0.5
# Randomize order of inputs for validation
x_perms = np.random.permutation(x_all.shape[0])
x_all = x_all[x_perms,:]
y_all = y_all[x_perms]
# Initialize the ImageClassifier
clf = ak.ImageClassifier(max_trials=40, objective = "val_accuracy", directory = "cifar10_checkpoints")
# Search for the best model
clf.fit(x_all, y_all, epochs=10, add_data = (x_noise, to_categorical(y_noise)))
# Evaluate on the testing data
print('Accuracy: {accuracy}'.format(accuracy=clf.evaluate(x_test, y_test)))