# coding: utf-8


import sys
from python_environment_check import check_packages
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:





d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'sklearn': '1.0',
}
check_packages(d)


# # Chapter 11 - Implementing a Multi-layer Artificial Neural Network from Scratch
# 

# ### Overview

# - [Modeling complex functions with artificial neural networks](#Modeling-complex-functions-with-artificial-neural-networks)
#   - [Single-layer neural network recap](#Single-layer-neural-network-recap)
#   - [Introducing the multi-layer neural network architecture](#Introducing-the-multi-layer-neural-network-architecture)
#   - [Activating a neural network via forward propagation](#Activating-a-neural-network-via-forward-propagation)
# - [Classifying handwritten digits](#Classifying-handwritten-digits)
#   - [Obtaining the MNIST dataset](#Obtaining-the-MNIST-dataset)
#   - [Implementing a multi-layer perceptron](#Implementing-a-multi-layer-perceptron)
#   - [Coding the neural network training loop](#Coding-the-neural-network-training-loop)
#   - [Evaluating the neural network performance](#Evaluating-the-neural-network-performance)
# - [Training an artificial neural network](#Training-an-artificial-neural-network)
#   - [Computing the loss function](#Computing-the-loss-function)
#   - [Developing your intuition for backpropagation](#Developing-your-intuition-for-backpropagation)
#   - [Training neural networks via backpropagation](#Training-neural-networks-via-backpropagation)
# - [Convergence in neural networks](#Convergence-in-neural-networks)
# - [Summary](#Summary)






# # Modeling complex functions with artificial neural networks

# ...

# ## Single-layer neural network recap






# ## Introducing the multi-layer neural network architecture










# ## Activating a neural network via forward propagation


# # Classifying handwritten digits

# ...

# ## Obtaining and preparing the MNIST dataset

# The MNIST dataset is publicly available at http://yann.lecun.com/exdb/mnist/ and consists of the following four parts:
# 
# - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, 60,000 examples)
# - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, 60,000 labels)
# - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, 10,000 examples)
# - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, 10,000 labels)
# 
# 





X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.astype(int).values

print(X.shape)
print(y.shape)


# Normalize to [-1, 1] range:



X = ((X / 255.) - .5) * 2


# Visualize the first digit of each class:




fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.savefig('figures/11_4.png', dpi=300)
plt.show()


# Visualize 25 different versions of "7":



fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X[y == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('figures/11_5.png', dpi=300)
plt.show()


# Split into training, validation, and test set:





X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)


# optional to free up some memory by deleting non-used arrays:
del X_temp, y_temp, X, y



# ## Implementing a multi-layer perceptron







##########################
### MODEL
##########################

def sigmoid(z):                                        
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):

    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        
        self.num_classes = num_classes
        
        # hidden
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # output
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
        
    def forward(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):  
    
        #########################
        ### Output layer weights
        #########################
        
        # onehot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use
        
        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out # "delta (rule) placeholder"

        # gradient for output weights
        
        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h
        
        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        

        #################################        
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        
        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out
        
        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative
        
        # [n_examples, n_features]
        d_z_h__d_w_h = x
        
        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, 
                d_loss__d_w_h, d_loss__d_b_h)




model = NeuralNetMLP(num_features=28*28,
                     num_hidden=50,
                     num_classes=10)


# ## Coding the neural network training loop

# Defining data loaders:




num_epochs = 50
minibatch_size = 100


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size 
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]

        
# iterate over training epochs
for i in range(num_epochs):

    # iterate over minibatches
    minibatch_gen = minibatch_generator(
        X_train, y_train, minibatch_size)
    
    for X_train_mini, y_train_mini in minibatch_gen:

        break
        
    break
    
print(X_train_mini.shape)
print(y_train_mini.shape)


# Defining a function to compute the loss and accuracy



def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)


def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets) 


_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)

predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)

print(f'Initial validation MSE: {mse:.1f}')
print(f'Initial validation accuracy: {acc*100:.1f}%')




def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
        
    for i, (features, targets) in enumerate(minibatch_gen):

        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        
        num_examples += targets.shape[0]
        mse += loss

    mse = mse/(i+1)
    acc = correct_pred/num_examples
    return mse, acc




mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc*100:.1f}%')




def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.1):
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):

        # iterate over minibatches
        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)

            #### Compute gradients ####
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini)

            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        
        #### Epoch Logging ####        
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc




np.random.seed(123) # for the training set shuffling

start_time = time.time()
epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid,
    num_epochs=50, learning_rate=0.1)
end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")


# ## Evaluating the neural network performance



plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel('Mean squared error')
plt.xlabel('Epoch')
plt.title('Training Loss Curve')
#plt.savefig('figures/11_07.png', dpi=300)
plt.show()




plt.plot(range(len(epoch_train_acc)), epoch_train_acc,
         label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc,
         label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy Over Epochs')
#plt.savefig('figures/11_08.png', dpi=300)
plt.show()




test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')
print(f'Test mse: {test_mse*100:.2f}%')





def compute_macro_auc(nnet, X, y, num_labels=10, minibatch_size=100):
    total_auc = 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        
        targets_binarized = label_binarize(targets, classes=np.arange(num_labels))
        
        auc = roc_auc_score(targets_binarized, probas, average='macro', multi_class='ovr')
        total_auc += auc
    
    macro_auc = total_auc / (i + 1)
    return macro_auc

test_macro_auc = compute_macro_auc(model, X_test, y_test)
print(f'Test Macro AUC: {test_macro_auc:.4f}')


# Plot failure cases:



X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]

_, probas = model.forward(X_test_subset)
test_pred = np.argmax(probas, axis=1)

misclassified_images = X_test_subset[y_test_subset != test_pred][:25]
misclassified_labels = test_pred[y_test_subset != test_pred][:25]
correct_labels = y_test_subset[y_test_subset != test_pred][:25]




fig, ax = plt.subplots(nrows=5, ncols=5, 
                       sharex=True, sharey=True, figsize=(8, 8))
ax = ax.flatten()
for i in range(25):
    img = misclassified_images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) '
                    f'True: {correct_labels[i]}\n'
                    f' Predicted: {misclassified_labels[i]}')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.savefig('figures/11_09.png', dpi=300)
plt.show()



# # Training an artificial neural network

# ...

# ## Computing the loss function






# ## Developing your intuition for backpropagation

# ...

# ## Training neural networks via backpropagation














# # Convergence in neural networks






# ...

# # Summary

# ...

# ---
# 
# Readers may ignore the next cell.







X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.astype(int).values

print(X.shape)
print(y.shape)




X = ((X / 255.) - .5) * 2




X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=123, stratify=y_temp)

del X_temp, y_temp, X, y


# # Two Hidden Layers Model




def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


class TowHiddenLayersNeuralNetMLP:
    def __init__(self, num_features=28*28, num_hidden1=500, num_hidden2=500, num_classes=10, random_seed=123):
        super().__init__()
        
        self.num_classes = num_classes
        
        rng = np.random.RandomState(random_seed)
        
        self.weight_h1 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden1, num_features))
        self.bias_h1 = np.zeros(num_hidden1)
        
        self.weight_h2 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden2, num_hidden1))
        self.bias_h2 = np.zeros(num_hidden2)
        
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden2))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h1 = np.dot(x, self.weight_h1.T) + self.bias_h1
        a_h1 = sigmoid(z_h1)
        
        z_h2 = np.dot(a_h1, self.weight_h2.T) + self.bias_h2
        a_h2 = sigmoid(z_h2)
        
        z_out = np.dot(a_h2, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        
        return a_h1, a_h2, a_out

    def backward(self, x, a_h1, a_h2, a_out, y):  
        y_onehot = int_to_onehot(y, self.num_classes)
        
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        
        d_z_out__dw_out = a_h2
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        
        d_z_out__a_h2 = self.weight_out
        d_loss__a_h2 = np.dot(delta_out, d_z_out__a_h2)
        d_a_h2__d_z_h2 = a_h2 * (1. - a_h2)
        d_z_h2__d_w_h2 = a_h1
        d_loss__d_w_h2 = np.dot((d_loss__a_h2 * d_a_h2__d_z_h2).T, d_z_h2__d_w_h2)
        d_loss__d_b_h2 = np.sum((d_loss__a_h2 * d_a_h2__d_z_h2), axis=0)
        
        d_z_h2__a_h1 = self.weight_h2
        d_loss__a_h1 = np.dot(d_loss__a_h2 * d_a_h2__d_z_h2, d_z_h2__a_h1)
        d_a_h1__d_z_h1 = a_h1 * (1. - a_h1)
        d_z_h1__d_w_h1 = x
        d_loss__d_w_h1 = np.dot((d_loss__a_h1 * d_a_h1__d_z_h1).T, d_z_h1__d_w_h1)
        d_loss__d_b_h1 = np.sum((d_loss__a_h1 * d_a_h1__d_z_h1), axis=0)

        return (d_loss__dw_out, d_loss__db_out, 
                d_loss__d_w_h2, d_loss__d_b_h2,
                d_loss__d_w_h1, d_loss__d_b_h1)

def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
        
    for i, (features, targets) in enumerate(minibatch_gen):

        _, _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        
        num_examples += targets.shape[0]
        mse += loss

    mse = mse/(i+1)
    acc = correct_pred/num_examples
    return mse, acc

def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.1):
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):
        
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            
            a_h1, a_h2, a_out = model.forward(X_train_mini)
            
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h2, d_loss__d_b_h2, d_loss__d_w_h1, d_loss__d_b_h1 = \
                model.backward(X_train_mini, a_h1, a_h2, a_out, y_train_mini)
            
            model.weight_h1 -= learning_rate * d_loss__d_w_h1
            model.bias_h1 -= learning_rate * d_loss__d_b_h1
            model.weight_h2 -= learning_rate * d_loss__d_w_h2
            model.bias_h2 -= learning_rate * d_loss__d_b_h2
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
              
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc

def compute_macro_auc(nnet, X, y, num_labels=10, minibatch_size=100):
    total_auc = 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    
    for i, (features, targets) in enumerate(minibatch_gen):
        _, _, probas = nnet.forward(features)
        
        targets_binarized = label_binarize(targets, classes=np.arange(num_labels))
        
        auc = roc_auc_score(targets_binarized, probas, average='macro', multi_class='ovr')
        total_auc += auc
    
    macro_auc = total_auc / (i + 1)
    return macro_auc

two_hidden_layers_model = TowHiddenLayersNeuralNetMLP(num_features=28*28, num_hidden1=500, num_hidden2=500, num_classes=10)




num_epochs = 50
minibatch_size = 100

np.random.seed(123)

start_time = time.time()
epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    two_hidden_layers_model, X_train, y_train, X_valid, y_valid, num_epochs=num_epochs, learning_rate=0.1)
end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")




plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel('Mean squared error')
plt.xlabel('Epoch')
plt.title('Training Loss Curve')
#plt.savefig('figures/11_07.png', dpi=300)
plt.show()




plt.plot(range(len(epoch_train_acc)), epoch_train_acc,
         label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc,
         label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy Over Epochs')
#plt.savefig('figures/11_08.png', dpi=300)
plt.show()




plt.show()
test_mse, test_acc = compute_mse_and_acc(two_hidden_layers_model, X_test, y_test)
test_macro_auc = compute_macro_auc(two_hidden_layers_model, X_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')
print(f'Test mse: {test_mse:.2f}')
print(f'Test Macro AUC: {test_macro_auc:.4f}')


# # Fully connected ANN implemented in Keras





class KerasModel:
    def __init__(self, input_shape=(28 * 28,), hidden_units=500, num_classes=10, learning_rate=0.1):
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))

        self.model.add(Dense(self.hidden_units, activation='sigmoid'))
        self.model.add(Dense(self.hidden_units, activation='sigmoid'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=SGD(learning_rate=learning_rate),
                           metrics=['accuracy'])

    def forward(self, x):
        z_h = np.dot(x, self.model.layers[0].get_weights()[0]) + self.model.layers[0].get_weights()[1]
        a_h = self.sigmoid(z_h)
    
        z_out = np.dot(a_h, self.model.layers[2].get_weights()[0]) + self.model.layers[2].get_weights()[1]
        a_out = self.softmax(z_out)
    
        return a_h, a_out


    def backward(self, x, a_h, a_out, y):
        y_onehot = np.zeros((y.shape[0], self.num_classes))
        y_onehot[np.arange(y.shape[0]), y] = 1
        
        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_z_out__a_h = self.model.layers[2].get_weights()[0]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        d_a_h__d_z_h = a_h * (1. - a_h)
        d_z_h__d_w_h = x
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def train(self, X_train, y_train_onehot, batch_size=100, epochs=20, validation_data=None):
        history = self.model.fit(X_train, y_train_onehot,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=validation_data,
                                 shuffle=True)
        return history

    def calculate_macro_auc(self, X_test, y_test_onehot):
        y_pred = self.model.predict(X_test)
        macro_auc = roc_auc_score(y_test_onehot, y_pred, average='macro', multi_class='ovr')
        return macro_auc

    def compute_mse_and_acc(self, X, y):
        y_pred = self.model.predict(X)
        y_onehot = np.zeros((y.shape[0], self.num_classes))
        y_onehot[np.arange(y.shape[0]), y] = 1
        mse = np.mean(np.square(y_pred - y_onehot))

        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = np.mean(np.equal(y_pred_class, np.argmax(y_onehot, axis=1)))

        return mse, accuracy

keras_model = KerasModel(input_shape=(28 * 28,), hidden_units=500, num_classes=10, learning_rate=0.1)




num_epochs = 50
minibatch_size = 100

np.random.seed(123)
y_train_onehot = int_to_onehot(y_train, 10)
y_valid_onehot = int_to_onehot(y_valid, 10)

start_time = time.time()
history = keras_model.train(X_train, y_train_onehot, batch_size=minibatch_size, epochs=num_epochs, validation_data=(X_valid, y_valid_onehot))
end_time = time.time()
total_time = end_time - start_time
print(f"Training completed in {total_time:.2f} seconds.")




plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.ylabel('Mean squared error')
plt.xlabel('Epoch')
plt.title('Training Loss Curve')
plt.show()




plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'],
         label='Training')
plt.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'],
         label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()




test_mse, test_acc = keras_model.compute_mse_and_acc(X_test, y_test)
test_macro_auc = keras_model.calculate_macro_auc(X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')
print(f'Test mse: {test_mse*100:.2f}%')
print(f"Test macro AUC: {test_macro_auc:.4f}")

