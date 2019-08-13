                     ###saver 保存读取###
import tensorflow as tf
import numpy as np

# ##Save to file
# #remember to define the same dtype and shape restore
# W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# init=tf.initialize_all_variables()
#
# saver=tf.train.Saver()
#
# with tf.Session()as sess:
#     sess.run(init)
#     save_path=saver.save(sess,"my_net/save_net.ckpt")
#     print("Save to path",save_path)


#restore variables
#redefine the same shape and same type for your variables才能正确导入

#定义一个空的框架
W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

#not need init step

saver=tf.train.Saver()

with tf.Session()as sess:
    saver.restore(sess,"my_net/save_net.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))

