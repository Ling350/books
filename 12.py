                         ###利用矩阵对身份证号（后四位）判断男女###
import tensorflow as tf
import random

random.seed()

x=tf.placeholder(tf.float32)
yTrain=tf.placeholder(tf.float32)

w1=tf.Variable(tf.random_normal([4,8],mean=0.5,stddev=0.1),dtype=tf.float32)
b1=tf.Variable(0,dtype=tf.float32)

xr=tf.reshape(x,[1,4])#调用tensorflow的reshape函数来把输入数据x从一个四维向量转换为一个形态为[1,4]的矩阵，并保存在变量xr中，
#后面隐藏层1的计算中也将用xr来进行计算。

n1=tf.nn.tanh(tf.matmul(xr,w1)+b1)

w2=tf.Variable(tf.random_normal([8,2],mean=0.5,stddev=0.1),dtype=tf.float32)
b2=tf.Variable(0,dtype=tf.float32)

n2=tf.matmul(n1,w2)+b2

y=tf.nn.softmax(tf.reshape(n2,[2]))#输出层对隐藏层2的输出n2进行softmax函数处理，使得结果是一个相加为1的向量。由于n2的形态
#[1,2]是一个矩阵，所以调用tf.reshape把它转换成了一个二维向量，与yTrain形式一样。

loss=tf.reduce_mean(tf.square(y-yTrain))

optimizer=tf.train.RMSPropOptimizer(0.01)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

lossSum=0

for i in range(50):
    xDataRandom=[int(random.random()*10),int(random.random()*10),int(random.random()*10),int(random.random()*10)]
    if xDataRandom[2]%2==0:
        yTrainDataRandom=[0,1]
    else:
        yTrainDataRandom=[1,0]

    result=sess.run([train,x,yTrain,y,loss],feed_dict={x:xDataRandom,yTrain:yTrainDataRandom})
    lossSum=lossSum+float(result[len(result)-1])

    print('i:%d,loss:%10.10f,avgLoss:%10.10f'%(i,float(result[len(result)-1]),lossSum/(i+1)))

