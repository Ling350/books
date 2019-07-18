                 ###用softmax函数来规范可变参数###
                #总分=德育分*0.6+智育分*0.3+体育分*0.1#
import tensorflow as tf

x=tf.placeholder(shape=[3],dtype=tf.float32)
yTrain=tf.placeholder(shape=[],dtype=tf.float32)

w=tf.Variable(tf.zeros([3]),dtype=tf.float32)

wn=tf.nn.softmax(w)
#这条语句中，新定义了一个变量wn，让其等于tf.nn.softmax(x)的返回值。nn是"neural network"的缩写，是tensorflow的一个重要子类（包）。
#其中的softmax函数是经常被用到的一个函数，他可以将一个向量规范化后得到一个新的向量，这个新的向量值加起来为1.softmax函数的这个特性
#经常被用来在神经网络中处理分类的问题，在这里，暂时只使用它来满足让所有权重值相加和为1的需求。这条语句执行完，会得到一个新的向量
#wn，它的各个值相加总和为1，并且它的形态与w完全相同。

n=x*wn

y=tf.reduce_sum(n)

loss=tf.abs(y-yTrain)

optimizer=tf.train.RMSPropOptimizer(0.1)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for i in range(2):
    result=sess.run([train,x,w,wn,y,yTrain,loss],feed_dict={x:[90,80,70],yTrain:85})
    print(result[3])#打印result下标为3的项：wn

    result=sess.run([train,x,w,wn,y,yTrain,loss],feed_dict={x:[98,95,87],yTrain:96})
    print(result[3])
