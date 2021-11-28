'''
- modiles 2021/11/22 @Dive
- model用到的一些通用模块 如 FM、Transformer
'''
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow

# pair构造group的fm交互
def pair_fm(group,num):
    fields=tf.reshape(group,
                      shape=[-1,num,self.emb_dim]) # [N,3,16]
    sum_square=tf.reduce_sum(fields, axis=1,keepdims=False) #[N,16]
    sum_square=tf.square(sum_square) # [N,16]
        
    square_sum=tf.square(fields) # [N,3,16]
    square_sum=tf.reduce_sum(square_sum,axis=1,keepdims=False) # [N,16]
        
    second_order=0.5*tf.subtract(sum_square,square_sum) # [N,16]
        
    return second_order

