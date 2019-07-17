         ###搭建解决三好学生成绩问题的神经网络w1=0.6,w2=0.3,w3=0.1###

import tensorflow as tf

#定义三个输入节点
x1=tf.placeholder(dtype=tf.float32)#dype(data type):占位符所代表的数值类型
x2=tf.placeholder(dtype=tf.float32)
x3=tf.placeholder(dtype=tf.float32)
#placeholder:占位符，即编写程序时还不确定输入什么数，而是在程序运行时才会输入，编程时仅仅把这个节点定义好，先‘占个位子’。
y_train=tf.placeholder(dtype=tf.float32)#真实值

#定义权重
w1=tf.Variable(0.1,dtype=tf.float32)#变量：神经网络的可变参数。0.1：初始值参数
w2=tf.Variable(0.1,dtype=tf.float32)
w3=tf.Variable(0.1,dtype=tf.float32)

#define 隐藏层三个节点
n1=x1*w1
n2=x2*w2
n3=x3*w3

#定义输出层
y=n1+n2+n3

loss=tf.abs(y-y_train)#误差
optimizer=tf.train.RMSPropOptimizer(0.001)#定义优化器
train=optimizer.minimize(loss) #按照loss最小化的原则来训练调整可变参数

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

#训练
for i in range(5000):
    result=sess.run([train,x1,x2,x3,w1,w2,w3,y,y_train,loss],feed_dict={x1:90,x2:80,x3:70,y_train:85})#以字典的形式喂入数据
    print(result)

    result=sess.run([train,x1,x2,x3,w1,w2,w3,y,y_train,loss],feed_dict={x1:98,x2:95,x3:87,y_train:96})#以字典的形式喂入数据
    print(result)