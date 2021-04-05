import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

# PART 1.1 Helper Functions
def relu(x):
    return (x * (x > 0))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x/np.sum(exp_x, axis=1, keepdims=True)

def compute(X, W, b):
    return (np.matmul(X,W) + b)

def averageCE(target, prediction):
    return (-1 * np.mean( target * np.log(prediction + 1e-12) ))

def gradCE(target, o):
    # derivation could be found in the report
    return (softmax(o) - target)

# PART 1.2 Gradients
def dL_by_dWo(target, o, h):
    # derivation could be found in the report, here the expression
    # is different than the report, in order to match the matrix size for the
    # multiplication. (i.e h: 10000xH --> hT:Hx10000)
    # Dimensions:
    # target, o : 10000x10
    # h         : 10000xH
    # output --> Hx10 (HxK)
    return np.matmul( np.transpose(h), gradCE(target, o) )

def dL_by_dbo(target, o):
    # derivation could be found in the report
    # target, o : 10000x10
    # one_matrix: 1x10000
    # output --> 1x10 (1xK)
    one_matrix = np.ones((1, target.shape[0]))
    return np.matmul(one_matrix, (gradCE(target, o)))

def dRelu(x):
    return ((x > 0) * 1)

def dL_by_dWh(target, o, x, hidden_input, Wo):
    # hidden input is WhX+bh
    # Wo is the output layer weight matrix (shape Hx10)
    # target and softmax(o) shape 1x10
    # input x is 10000x784 --> xT: 784x10000
    # hidden_input : 10000xH
    # gradCE : 10000x10
    # Wo: Hx10
    # output --> 784xH
    # this is to match the dimensions, since the relu is dependent on 
    # the number of nodes, not the input.
    # this is equivalent to summing over K output nodes
    return np.matmul(np.transpose(x), \
            (dRelu(hidden_input) \
            * np.matmul((gradCE(target, o)), np.transpose(Wo))))

def dL_by_dbh(target, o, hidden_input, Wo):
    # hidden input is WhX+bh
    # Wo is the output layer weight matrix (shape Hx10)
    # target and softmax(o) shape 1x10
    # input x is 10000x784 --> xT: 784x10000
    # hidden_input : 10000xH
    # one_matrix = 1x10000
    # gradCE : 10000x10
    # Wo: Hx10
    # output -->1x10000 * 10000xH = 1xH
    one_matrix = np.ones((1, hidden_input.shape[0]))
    # this is to match the dimensions, since the relu is dependent on
    # the number of nodes, not the input.
    # this is equivalent to summing over K output nodes
    return np.matmul(one_matrix, \
            (dRelu(hidden_input) \
            * np.matmul((gradCE(target, o)), np.transpose(Wo))))

# PART 1.3 Learning
def forward_propagation(x, W_h, b_h, W_o, b_o, target):
    hidden_input = np.add(np.matmul(x, W_h), b_h)
    h = relu(hidden_input)
    o = np.add(np.matmul(h, W_o), b_o)
    p = softmax(o)
    return p, o, h, hidden_input

def train(train_data, train_target, val_data, val_target, F, H, K, \
          Wo, Wh, bo, bh, vWo, vWh, vbo, vbh, gamma, alpha, epochs):
  
    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = []
    
    for _ in range(epochs):
      # forward pass for training
      prediction, o, h, hidden_input = forward_propagation(train_data, Wh, bh, Wo, bo, train_target)
      # calculate loss and accuracy
      loss.append(averageCE(train_target, prediction))
      prediction_idx  = prediction.argmax(axis = 1) 
      target_idx = train_target.argmax(axis=1)
      check_equal = (prediction_idx==target_idx)
      accuracy.append(np.mean(check_equal))

      # forward pass for validation
      val_prediction, val_o, val_h, val_hidden_input = forward_propagation(val_data, Wh, bh, Wo, bo, val_target)
      # calculate loss and accuracy
      val_loss.append(averageCE(val_target, val_prediction))
      val_prediction_idx  = val_prediction.argmax(axis = 1) 
      val_target_idx = val_target.argmax(axis=1)
      val_check_equal = (val_prediction_idx == val_target_idx)
      val_accuracy.append(np.mean(val_check_equal))

      # calculate gradients
      dL_dWo = dL_by_dWo(train_target, o, h)
      dL_dbo = dL_by_dbo(train_target, o)
      dL_dWh = dL_by_dWh(train_target, o, train_data, hidden_input, Wo)
      dL_dbh = dL_by_dbh(train_target, o, hidden_input, Wo)

      # update weights and biases
      vWh = (gamma * vWh) + (alpha * dL_dWh)
      vWo = (gamma * vWo) + (alpha * dL_dWo)
      vbh = (gamma * vbh) + (alpha * dL_dbh)
      vbo = (gamma * vbo) + (alpha * dL_dbo)
      Wh = Wh - vWh
      Wo = Wo - vWo
      bh = bh - vbh
      bo = bo - vbo

    return Wh, bh, Wo, bo, accuracy, loss, val_accuracy, val_loss

def numpy_main():
    F, H, K = 784, 1000, 10
    epochs = 200
    gamma = 0.99
    alpha = 2 * 1e-7

    # Initialize
    mean = 0
    variance_out = 2/(H+K)
    variance_hidden = 2/(F+K)
    std_dev_out = np.sqrt(variance_out)
    std_dev_hidden = np.sqrt(variance_hidden)
    Wo = np.random.normal(mean, std_dev_out, (H,K))
    Wh = np.random.normal(mean, std_dev_hidden, (F,H))
    bo = np.zeros((1, K))
    bh = np.zeros((1, H))
    vWo = np.full((H,K), 1e-5)
    vWh = np.full((F,H), 1e-5)
    vbo = np.zeros((1, K))
    vbh = np.zeros((1, H))

    train_data, valid_data, test_data, train_target, valid_target, test_target = loadData()
    train_data = train_data.reshape(np.shape(train_data)[0], np.shape(train_data)[1]*np.shape(train_data)[2])
    valid_data = valid_data.reshape(np.shape(valid_data)[0], np.shape(valid_data)[1]*np.shape(valid_data)[2]) 
    test_data = test_data.reshape(np.shape(test_data)[0], np.shape(test_data)[1]*np.shape(test_data)[2])

    # # This is only for testing with small batches, uncomment to get faster results
    # batch_size = 500
    # batch_start = 0
    # random_indexes = np.random.permutation(len(train_data))
    # train_data_shuffled = train_data[random_indexes]
    # train_target_shuffled = train_target[random_indexes]
    # train_batch = train_data_shuffled[batch_start:batch_start + batch_size, :]
    # target_batch = train_target_shuffled[batch_start:batch_start + batch_size]
    # train_data = train_batch
    # train_target = target_batch

    new_train, new_valid, new_test = convertOneHot(train_target, valid_target, test_target)
    W_h, b_h, W_o, b_o, accuracy, loss, val_accuracy, val_loss = train(train_data, new_train, valid_data, new_valid, F, H, K,\
                                               Wo, Wh, bo, bh, vWo, vWh, vbo, vbh, gamma, alpha, epochs)
  
    # calculate the final accuracy (test set)
    test_prediction, test_o, test_h, test_hidden_input = forward_propagation(test_data, W_h, b_h, W_o, b_o, new_test)
    # calculate loss and accuracy
    test_prediction_idx = test_prediction.argmax(axis = 1) 
    test_target_idx = new_test.argmax(axis = 1)
    test_check_equal = (test_prediction_idx == test_target_idx)
    test_accuracy = (np.mean(test_check_equal))

    epochs_plot = range(epochs)
    plt.plot(epochs_plot, loss, 'g', label='Training loss')
    plt.plot(epochs_plot, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(epochs_plot, accuracy, 'g', label='Training acc.')
    plt.plot(epochs_plot, val_accuracy, 'b', label='Validation acc.')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    print ("Training accuracy = ", str(accuracy[len(accuracy) - 1]))
    print ("Validation accuracy = ", str(val_accuracy[len(val_accuracy) - 1]))
    print ("Test accuracy = ", str(test_accuracy))

if __name__ == "__main__":
  numpy_main()
