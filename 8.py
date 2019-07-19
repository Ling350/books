                     ###成绩二分类问题(使用随机产生的数据)###

import tensorflow as tf
import random

random.seed()

x=tf.placeholder(dtype=tf.float32)
yTrain=tf.placeholder(dtype=tf.float32)

w=tf.Variable(tf.zeros([3]),dtype=tf.float32)
b=tf.Variable(80,dtype=tf.float32)


n1=w*x

n2=tf.reduce_sum(n1)-b

y=tf.nn.sigmoid(n2)

loss=tf.abs(yTrain-y)

optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for i in range(5):
    xData=[int(random.random()*8+93),int(random.random()*8+93),int(random.random()*8+93)]
    xAll=xData[0]*0.6+xData[1]*0.3+xData[2]*0.1
    if xAll>=95:
        yTrainData=1
    else:
        yTrainData=0

        result=sess.run([train,x,yTrain,w,b,n2,y,loss],feed_dict={x:xData,yTrain:yTrainData})
        print(result)

    xData=[int(random.random()*41+60),int(random.random()*41+60),int(random.random()*41+60)]
    xAll=xData[0]*0.6+xData[1]*0.3+xData[2]*0.1
    if xAll>=95:
        yTrainData=1
    else:
        yTrainData=0
        result=sess.run([train,x,yTrain,w,b,n2,y,loss],feed_dict={x:xData,yTrain:yTrainData})

        print(result)




