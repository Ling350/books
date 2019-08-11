                     ###用Keras实现神经网络模型###

import tensorflow.contrib.keras as k
import random
import numpy as np

random.seed()

rowSize=4
rowCount=8192

#生成随机训练数据

xDataRandom=np.full((rowCount,rowSize),5,dtype=np.float32)#创建一个由常数填充的数组,第一个参数是数组的形状(8192*4)，
                                                          # 第二个参数是数组中填充的常数
yTrainDataRandom=np.full((rowCount,2),0,dtype=np.float32) #(8192*2)
for i in range(rowCount):
    for j in range(rowSize):
        xDataRandom[i][j]=np.floor(random.random()*10)#np.floor():返回数字的下舍整数.random.random()生成0和1之间的随机浮点数
        if xDataRandom[i][2]%2==0: #第三列是否为偶数
            yTrainDataRandom[i][0]=0
            yTrainDataRandom[i][1]=1
        else:
            yTrainDataRandom[i][0]=1
            yTrainDataRandom[i][1]=0


model=k.models.Sequential()  #定义一个model变量，调用k.models.Sequential函数生成了一个顺序化的模型。我们的模型至今为止都是
                             #顺序化的模型，也就是一层连着一层顺序排列的模型

##分别定义了3个全连接层并顺序增加到模型中。定义全连接层是用k.layers.Dense函数来实现。
##该函数的第一个参数代表该层准备输出的维度，即本层有多少个神经元节点
##命名参数input_dim用于指定本层输入的维度（也就是上一层输出节点的个数）
##命名参数activation用于指定本层使用的激活函数。注:第3层中把softmax作为该层的激活函数，而不是像以前的代码中那样放到输出层，效果一样。
model.add(k.layers.Dense(32,input_dim=4,activation='tanh'))  ##tanh(x*w+b)

model.add(k.layers.Dense(32,input_dim=32,activation='sigmoid'))

model.add(k.layers.Dense(2,input_dim=32,activation='softmax'))

##用model的compile函数来设置误差函数和优化器等，制定了loss函数使用mean_squared_error，也就是均方差；
##优化器使用RMSProp
##metrics指定训练的指标，一般我们会指定['accuracy']，即精确度。
model.compile(loss='mean_squared_error',optimizer='RMSProp',metrics=['accuracy'])

##用Keras模型的fit函数就可以进行训练。fit函数中
##第一个参数输入的训练数据集，在这里传入xDataRandom这个我们生成的一批训练数据，是一个二维数组；
##第二个参数输入的目标值，我们传入yTrainDataRandom这个与xDataRandom每行数据一一对应的目标值结果
##命名参数epochs用于指定训练多少轮次
##命名参数batch_size用于指定训练多少批次后进行可变参数调节的梯度更新，这主要影响可变参数的调整速度，因此会影响整个训练过程的速度，默认值32
##命名参数verbose用于指定Keras在训练过程中输出信息的频繁程度，0代表最少的输出信息，一般用2代表尽量多一点输出信息
model.fit(xDataRandom,yTrainDataRandom,epochs=10,batch_size=1,verbose=2)#batch_size:一次训练所选取的样本数
##例：训练集有1000个样本，batchsize=10，那么： 训练完整个样本集需要： 100次iteration，1次epoch



