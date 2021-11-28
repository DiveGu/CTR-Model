import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 测试 [N,H1,H0,k] [H1,H0] 要得 [N,k] 怎么算
# H0 H1 H2 = 6,7,8

# X=np.arange(1,5*7*6*10+1).reshape((5,7,6,10))
# X=tf.cast(X,tf.float32)
# Y=np.arange(1,7*6+1).reshape((7,6))
# Y=tf.cast(Y,tf.float32)
# print(X.shape) # [5,7,6,10]
# print(Y.shape) # [7,6]
# # print(X.get_shape().as_list())

# X_t=tf.transpose(X,perm=[0,3,1,2])

# Z=X_t*Y # [5,10,7,6] [7,6]
# print(Z.shape) # [5,10,7,6]

# Z=tf.reduce_sum(Z,axis=3,keepdims=False)
# Z=tf.reduce_sum(Z,axis=2,keepdims=False)
# print(Z.shape)

# with tf.Session() as sess:
#     ret = sess.run(Z)

# print(ret)

# ====================================================
# # 这种写法不对
# # 测试 [N,H1,H0,k] [H1,H0] 要得 [N,k] 怎么算
# # H0 H1 H2 = 6,7,8

# X=np.arange(1,5*7*6*10+1).reshape((5,7,6,10))
# X=tf.cast(X,tf.float32)
# Y=np.arange(1,7*6+1).reshape((7,6))
# Y=tf.cast(Y,tf.float32)
# print(X.shape) # [5,7,6,10]
# print(Y.shape) # [7,6]

# X_t=tf.transpose(X,perm=[0,3,2,1]) # [5,10,6,7]

# Z=tf.matmul(X_t,Y) # [5,10,6,7] [7,6]
# print(Z.shape) # [5,10,6,6]


# with tf.Session() as sess:
#     ret = sess.run(Z)

# print(ret)


# # 测试 a//b b为小数的情况
# # 3//0.1得到 29.0
# a=3
# b=0.1
# c=a//b

# print(c)

# 测试 [N,2,6] * [N,2] 不能用广播
# 必须弄成 [N,2,6] * [N,2,1]

X=np.arange(1,36+1).reshape((3,2,6))
X=tf.cast(X,tf.float32)
Y=np.arange(1,6+1).reshape((3,2))
Y=tf.cast(Y,tf.float32)

# Z=X*Y # 报错
Z=X*tf.expand_dims(Y,axis=2) # [N,2,6] * [N,2,1]
print(Z.shape) # [N,2,6] * [N,2,6]


with tf.Session() as sess:
    ret = sess.run(Z)

print(ret)


