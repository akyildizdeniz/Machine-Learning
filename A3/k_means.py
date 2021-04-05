import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    x_expanded = tf.expand_dims(X, 0) # dimension = (1, N, D)
    mu_expanded = tf.expand_dims(MU, 1) # dimension = (K, 1, D)
    # tf.subtract is defined to subtract the last dimensions
    # for all the other dimensions, so by default it works for us
    dist = tf.reduce_sum(tf.square(tf.subtract(x_expanded, mu_expanded)), 2)
    pair_dist = tf.transpose(dist)
    return pair_dist

def nearest_centroid(X, MU):
    # Return nearest centroids
    distances = distanceFunc(X, MU)
    # distances is NxK tensor, getting the index
    # of the minimum argument in axis=1 will give
    # us the value of the closest centroid.
    nearest_centroids = tf.argmin(distances, 1)
    return nearest_centroids

def buildGraph(beta1 = None, beta2 = None, epsilon = None, alpha = None, K = None, D = None, N = None):
    X = tf.placeholder(tf.float32, [None, D], name ='X')
    MU_tmp = tf.truncated_normal([K,D], stddev=0.25)
    MU = tf.Variable(MU_tmp, dtype = tf.float32)
    pair_dist = distanceFunc(X,MU)
    loss = tf.reduce_sum(tf.reduce_min(pair_dist, axis = 1))
    optimizer = tf.train.AdamOptimizer(learning_rate = alpha, beta1 = beta1, beta2 = beta2, epsilon = epsilon)
    training_op = optimizer.minimize(loss=loss)
    return X, MU, pair_dist, loss, optimizer, training_op

def k_means(is_valid):
  # Loading data
  data = np.load('data2D.npy')
  # data = np.load('data100D.npy')
  [num_pts, dim] = np.shape(data)

  # For Validation set
  if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]

  epochs = 200
  N = num_pts
  D = dim
  K = 1
  X, MU, distance, loss_var, optimizer, train = buildGraph(beta1 = 0.9, beta2 = 0.99, epsilon = 1e-5, alpha = 0.1, K = K, D = D, N = N)


  loss_array = []
  val_loss_array = []
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_op)
    for step in range(epochs):
      feed_dict={X:data}
      _, center, loss, _ = sess.run([X, MU, loss_var, train], feed_dict=feed_dict)
      loss_array.append(loss)
      if is_valid:
        feed_dict = {X:val_data}
        val_center, val_loss, _ = sess.run([MU, loss_var, train], feed_dict = feed_dict)
        val_loss_array.append(val_loss)
    # Now, get the nearest centroids for plotting, going to be the color code
    feed_dict = {X:data, MU:center}
    nearest_centroids = sess.run(nearest_centroid(X, MU), feed_dict = feed_dict)


  percentage = np.bincount(nearest_centroids)/(nearest_centroids.size)
  epochs_plot = range(epochs)
  plt.plot(epochs_plot, loss_array, 'g', label='Training loss')
  if is_valid:
    plt.title('Training loss plot for K ='+str(K) +' \n Final validation loss = ' + str(val_loss_array[len(val_loss_array) - 1]))
  else: 
    plt.title('Training loss plot for K =' +str(K))
  plt.xlabel('Updates')
  plt.ylabel('Loss')
  plt.show()

  # This did not work
  # for i in range(len(percentage)):
  #   plt.scatter(data[:, 0], data[:, 1], c = nearest_centroids, s = 5, alpha = 1, label=f'{percentage[i]:.2f} %')

  plt.scatter(data[:, 0], data[:, 1], c = nearest_centroids, s = 5, alpha = 1)
  plt.title('Scatters for K = '+str(K))
  plt.xlabel('x1')
  plt.ylabel('x2')
  # plt.legend()
  plt.show()

  sum = 0
  for i in range(len(percentage)):
    print("Percentage of class " ,i, " is: ", str(percentage[i]*100))
    sum += percentage[i]*100
  # Could print this to make sure they add up to 100
  # print(sum)

if __name__ == "__main__":
  is_valid = True
  k_means(is_valid)
