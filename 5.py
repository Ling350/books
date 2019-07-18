                ###Tensorflow会很智能地在程序执行时根据输入的数据自动确定张量的形态###
import tensorflow as tf

x=tf.placeholder(dtype=tf.float32)
xshape=tf.shape(x)

sess=tf.Session()

result=sess.run(xshape,feed_dict={x:8})
print(result)

result=sess.run(xshape,feed_dict={x:[1,2,3]})
print(result)

result=sess.run(xshape,feed_dict={x:[[1,2,3],[3,6,9]]})
print(result)