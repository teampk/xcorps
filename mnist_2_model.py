import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
import tensorflow as tf

# -------------1-1. 이미지 => 숫자형 데이터 (train)---------------------------------------
# Directory 경로 호출
TRAIN_DIR = '/Users/paeng/dev/data/MNIST/trainingSet/'
# ['0_zero' '9_nine' '6_six' '4_four' '2_two' '7_seven' '5_five' '1_one' '3_three' '8_eight']
train_folder_list = array(os.listdir(TRAIN_DIR))
train_input = []
train_label = []
 
label_encoder = LabelEncoder()  # LabelEncoder Class 호출

# [0 1 2 3 4 5 6 7 8 9]
integer_encoded = label_encoder.fit_transform(train_folder_list)
onehot_encoder = OneHotEncoder(sparse=False) 

# [[0] [1] [2] [3] [4] [5] [6] [7] [8] [9]]
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# index -> 0 ~ 9
for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    # ['img_1.jpg' 'img_3.jpg' 'img_10.jpg' ... 'img_n.jpg']
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        # Gray Scaling
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img를 np.array 형식으로 붙여넣기 (.append는 R에서의 rbind같은 명령문)
        train_input.append([np.array(img)])
        train_label.append([np.array(onehot_encoded[index])])
# -1 -> 데이터 개수를 정확히 모를 때 // 784 -> 28*28 image size
# train_input의 shape 출력시 (42000, 784) 즉 42000개의 데이터와 784개의 숫자
train_input = np.reshape(train_input, (-1, 784))
# train_input의 shape 출력시 (42000, 10)
train_label = np.reshape(train_label, (-1, 10))

# data type을 float32로 변환
train_input = np.array(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)
# 최종적으로 저장
np.save("train_data.npy", train_input)
np.save("train_label.npy", train_label)


# -------------1-2. 이미지 => 숫자형 데이터 (test)---------------------------------------
# Directory 경로 호출
TEST_DIR = '/Users/paeng/dev/data/MNIST/testSet/'
test_folder_list = array(os.listdir(TEST_DIR))
 
test_input = []
test_label = []
 
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(test_folder_list)
 
onehot_encoder = OneHotEncoder(sparse=False) 
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
 
for index in range(len(test_folder_list)):
    path = os.path.join(TEST_DIR, test_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_input.append([np.array(img)])
        test_label.append([np.array(onehot_encoded[index])])
 
test_input = np.reshape(test_input, (-1, 784))
test_label = np.reshape(test_label, (-1, 10))
test_input = np.array(test_input).astype(np.float32)
test_label = np.array(test_label).astype(np.float32)
np.save("test_input.npy",test_input)
np.save("test_label.npy",test_label)

# -------------2. 모델 설계---------------------------------------
# hyper parameters
learning_rate = 0.001
# input place holders
X = tf.placeholder(tf.float32, [None, 784])
# batch size = -1 (가변적) / channel 수 = 1 (RGB는 3)
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28 x 28 x 1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

print('---- Conv Layer 1 ----')
# 3 * 3 필터 / 1 = input data의 channel / 32 = 필터를 32개 쓸 것이다. / stddev는 난수의 변동 (알 필요 x)
# < 28 * 28 * 1 >
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# Convolution Layer 선언. W1 필터 적용 / 1 1 1 1 모든 batch와 channel에 대해 convolution 적용, 우측 1칸 아래로 1칸
# padding = 'SAME' -> convolution 연산 후 shape 줄어드는 것 방지. 
# padding = 'VALID' -> 28*28 => 26*26 이 된다. 
# < 28 * 28 * 1 >
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
print('conv1', L1)
# < 28 * 28 * 32 > (conv)
# RELU Activation Function
L1 = tf.nn.relu(L1)
print('conv1', L1)
# max-pooling layer / ksize = kernel size / 1 2 2 1 모든 batch와 channel에 대해 2*2 크기의 kernel size / 
# < 28 * 28 * 32 >
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('conv1', L1)
# < 14 * 14 * 32 > (max pooling)

print('---- Conv Layer 2 ----')
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
print('conv2', L2)
# < 14 * 14 * 64 > (conv)
L2 = tf.nn.relu(L2)
print('conv2', L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('conv2', L2)
# < 7 * 7 * 64 > (max pooling)


print('---- Fully Connected Layer ----')
# Fully-Connected 로 넣기 위해서 쭉 펼친 형태로 Reshape을 한다.
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
print('reshaped', L2_flat)
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

# define cost/loss optimizer (Adam Optimizer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


'''
# -------------3. 모델 학습--------------------------------------------
# 학습 횟수
training_epochs = 15
# 한번 학습하는데 100개의 데이터를 사용.
batch_size = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('===Learning Started===')
for epoch in range(training_epochs):
    avg_cost = 0.0
    # 42000개의 데이터이므로 total_batch 는 420
    total_batch = int(len(train_input / batch_size))

    for i in range(total_batch):
        start = ((i+1) * batch_size) - batch_size
        end = ((i+1)*batch_size)
        batch_xs = train_input[start:end]
        batch_ys = train_label[start:end]
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
print('===Learning Finished===')

# -------------4. 모델 정확도 산출---------------------------------------
# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: test_input, Y: test_label}))
'''