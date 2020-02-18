import tensorflow as tf
import numpy as np
import pandas as pd
import random

learning_rate = 0.001
training_epochs = 15
batch_size = 100

data = pd.read_csv('fashion-mnist_train.csv', sep = ',')  
xy_train_data = []
for i in range(60000):
    xy_train_data.append(list(data.iloc[i]))
# y_train = list(data['label'])

"""x_train, y_train shape == (60000, 784) (60000,)"""
#################################################################################

X = tf.placeholder(tf.float32, shape = [None, 784])
X_img = tf.reshape(X, [-1, 28,28, 1]) 
Y = tf.placeholder(tf.float32, shape = [None, 1])

# conv > relu > maxpool
#layer 1

Filter1 = tf.Variable(tf.truncated_normal(shape = [3,3,1,32], stddev = 0.01))
conv1 = tf.nn.conv2d(X_img, Filter1, strides = [1,1,1,1], padding = 'SAME') 
Activation1 = tf.nn.relu(conv1)
M_pool1 = tf.nn.max_pool(Activation1, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
# 14x14x

# layer 2
Filter2 = tf.Variable(tf.truncated_normal(shape = [3,3,32,64], stddev = 0.01))
conv2 = tf.nn.conv2d(M_pool1, Filter2, strides = [1,1,1,1], padding = 'SAME' )
Activation2 = tf.nn.relu(conv2)
M_pool2 = tf.nn.max_pool(Activation2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# 7*7*8  Full layer

FC = tf.reshape(M_pool2, [-1,7*7*64])
W_1 = tf.Variable(tf.random_normal([7*7*64, 10]))
B = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(FC, W_1) + B # None x 10 
 
logits = []
for i in range(batch_size):
    logits.append([np.argmax(hypothesis[i], axis = 0)])
logits = np.reshape(logits, [100,1])

# >>>> 10 개의 y 값 
 # argmax 이용해서 one-hot 값 뽑아낸 후 label 이랑 비교하기 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Session open

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = float(60000 / batch_size)
    #total_batch = 600
    for i in range(total_batch):
            Random = random.sample(xy_train_data, batch_size)
            x_train = []
            y_train = []
            for j in range(batch_size):
                x_train.append(Random[j][1:])
            X_i = x_train  # (100, 784)
            
            for k in range(batch_size):
                y_train.append(Random[k][0])
            Y_i = y_train  # (100,1)
            
            # Y_i : (100,1)  >>>  Y_placeholder = (None, 10)   > 
            feed_dict = {X : X_i , Y : Y_i}
            c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
            avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print("Learning finished!")        
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y , 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
########################## test accuracy 
# print('accuracy:', sess.run(accuracy, feed_dict ={X:x_test} , Y: y_train))
