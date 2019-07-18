                                     ###将2中的输入改为数组的形式###
import tensorflow as tf

x=tf.placeholder(shape=[3],dtype=tf.float32)#shape:表示变量x的形态，取值为3，表示为x的数据将是一个有3个数字的数组，
#也就是一个三维向量
yTrain=tf.placeholder(shape=[],dtype=tf.float32)#yTrain因为只是一个普通数字，不是向量，如果给他一个形态的话，可以用一个空的方括号[]代表

w=tf.Variable(tf.zeros([3],dtype=tf.float32))#返回值将是一个数组[0,0,0],这个向量将作为w的初始值

n=x*w#假设输入数据x为[90,80,70],也就是一位学生的3项分数，此时w为[2,3,4]，那么n=x*w的运算结果就是[90*2,80*3,70*4]
#即[180,240,280].因为*代表数学中矩阵运算的“点乘”，点乘是指两个形态相同的矩阵中每个相同位置的数字相乘。

y=tf.reduce_sum(n)#tf.reduce_sum函数的作用是把作为它的参数向量（以后还可能是矩阵）中的所有维度的值相加求和

loss=tf.abs(y-yTrain)

optimizer=tf.train.RMSPropOptimizer(0.001)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for i in range(5000):
    result=sess.run([train,x,w,y,yTrain,loss],feed_dict={x:[90,80,70],yTrain:85})
    print(result)

    result=sess.run([train,x,w,y,yTrain,loss],feed_dict={x:[98,95,87],yTrain:96})
    print(result)



