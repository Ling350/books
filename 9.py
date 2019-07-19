                           ###批量随机生成训练数据###
import tensorflow as tf
import random
import numpy as np

random.seed()

rowCount=5

xData=np.full(shape=(rowCount,3),fill_value=0,dtype=np.float32)#np.full():生成多维数组，用函数np.full生成形态为(rowCount,3)
#的多维数组，并全部用零来填充，参数fill_value用于把这个数组中的所有值预先填充为某个数。注意数据类型
yTrainData=np.full(shape=rowCount,fill_value=0,dtype=np.float32)


#for循环，批量生成5条训练数据
for i in range(rowCount):
    xData[i][0]=int(random.random()*11+90)
    xData[i][1]=int(random.random()*11+90)
    xData[i][2]=int(random.random()*11+90)

    xAll=xData[i][0]*0.6+xData[i][1]*0.3+xData[i][2]*0.1
    if xAll>=95:
        yTrainData[i]=1
    else:
        yTrainData [i]=0

print('xData:%s'%xData)
print('yTrainData:%s'%yTrainData)


x=tf.placeholder(dtype=tf.float32)
yTrain=tf.placeholder(dtype=tf.float32)

w=tf.Variable(tf.zeros([3]),dtype=tf.float32)
b=tf.Variable(80,dtype=tf.float32)

wn=tf.nn.softmax(w)

n1=wn*x

n2=tf.reduce_sum(n1)-b

y=tf.nn.sigmoid(n2)

loss=tf.abs(yTrain-y)

optimizer=tf.train.GradientDescentOptimizer(0.1)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for i in range(500):
    for j in range(rowCount):
        result=sess.run([train,x,yTrain,wn,b,n2,y,loss],feed_dict={x:xData[j],yTrain:yTrainData[j]})
        print(result)