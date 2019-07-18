                      ###为了严格规范输入数据，在定义张量的时候，就可以指定形态，例如：###
import tensorflow as tf

x=tf.placeholder(shape=[2,3],dtype=tf.float32)

xshape=tf.shape(x)

sess=tf.Session()

result=sess.run(xshape,feed_dict={x:[[1,2,3],[2,4,6]]})
print(result)

#故意错误的形式
#result=sess.run(xshape,feed_dict={x:[1,2,3]})
#print(result)
