import tensorflow as tf
#tensorflow를 사용하기 위해서 import해옴.
from tensorflow.examples.tutorials.mnist import input_data
#데이터를 자동으로 다운로드하고 설치하는 코드.

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
#데이터를 자동으로 다운로드하고 설치하는 코드.

# Set up model
x = tf.placeholder(tf.float32, [None, 784])
#부정소숫점으로 이루어진 2차원 텐서로, 각 이미지들은 784차원의 벡터로 단조화되어있음.
W = tf.Variable(tf.zeros([784, 10]))
#784차원의 이미지 벡터를 곱하여 10차원벡터의 증거를 만든다는 뜻.
b = tf.Variable(tf.zeros([10]))
#[10]의 형태이므로 출력에 더할 수 있습니다.
y = tf.nn.softmax(tf.matmul(x, W) + b)
#x와 W를 곱한다음에,(결과가 뒤집힌 이유는 2d텐서일 경우를 다룰 수 있게 하기 위한 잔재주임). 그다음b를 더함.
y_ = tf.placeholder(tf.float32, [None, 10])
#교차 엔트로피를 구현하기위해 새로운placeholder를 추가함.

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#그 다음 교차엔트로피를 구함.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#학습도를 0.01로 준 경사 하강법 알고리즘을 이용하여 교차 엔트로피를 최소화하도록 명령.

# Session
init = tf.initialize_all_variables()
#실행 전 마지막으로 만들었던 변수들을 초기화함.

sess = tf.Session()
sess.run(init)
#세션에서 모델을 시작하고 변수들을 초기화하는 작업을 실행할 수 있습니다.

# Learning
for i in range(1000):
#학습을 1000번 시켜봄.
  batch_xs, batch_ys = mnist.train.next_batch(100)
#각 반복단계마다 학습세트로부터 100개의 랜덤데이터들의 일괄처리를 가져옴.
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#placeholders를 대체하기 위한 일괄 처리 데이터에 train_step피딩을 실행함.

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))