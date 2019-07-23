                                     ###身份证（后四位）来判断男女###
import tensorflow as tf
import random

random.seed()

x=tf.placeholder(tf.float32)
yTrain=tf.placeholder(tf.float32)

w=tf.Variable(tf.random_normal([4],mean=0.5,stddev=0.1),dtype=tf.float32)#产生符合正态分布的一个四维向量
b=tf.Variable(0,dtype=tf.float32)

n1=w*x+b

y=tf.nn.sigmoid(tf.reduce_sum(n1))

loss=tf.abs(y-yTrain)

optimizer=tf.train.RMSPropOptimizer(0.01)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

lossSum=0

for i in range(100):
    xDataRandom=[int(random.random()*10),int(random.random()*10),int(random.random()*10),int(random.random()*10)]
    if xDataRandom[2]%2==0:
        yTrainDataRandom=0
    else:
        yTrainDataRandom=1

    result=sess.run([train,x,yTrain,y,loss],feed_dict={x:xDataRandom,yTrain:yTrainDataRandom})
    lossSum=lossSum+float(result[len(result)-1])

    print('i:%d,loss:%f,avgLoss:%f'%(i,float(result[len(result)-1]),lossSum/(i+1)))


