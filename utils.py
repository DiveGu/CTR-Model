'''
- utils 2021/11/22 @Dive
- 各种工具函数
    - 训练模型
    - 预测模型
'''
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow
import time
import argparse
import os

from sklearn.metrics import roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# 加载训练数据 测试数据
def get_batch_data(array,batch_size,idx):
    start=idx*batch_size
    end=(idx+1)*batch_size
    end=end if end<=array.shape[0] else array.shape[0]
    return array[start:end]

# (key:model.placeholder,value:array)
def get_feed_dict(model,array,train_flag):
        # input_dict 和 array的所有特征必须按照顺序来
        i=0
        cols_cnt=array.shape[1]
        feed_dict=dict()
        # 所有的 X y
        for k,v in model.input_dict.items():
            feed_dict[model.input_dict[k]]=array[:,i]
            i+=1
            if(i==cols_cnt):
                break

        if(train_flag):
            feed_dict[model.input_dict['dnn_keep_prob']]=1.0
        else:
            feed_dict[model.input_dict['dnn_keep_prob']]=1.0
        
        return feed_dict

# 改成并行
def get_batch_feed_dict(df,batch_size,cols,y,idx):
    tmp=get_batch_data(df,batch_size,idx)
    return convert_feed_dict(tmp,cols,y)
    

# 按照batch_size对X进行预测
def predict_by_batch(model,sess,array,batch_size):
    predict=[]
    n_batch=array.shape[0]//batch_size+1
    for idx in range(n_batch):
        batch=get_batch_data(array,batch_size,idx)
        feed_dict=get_feed_dict(model,batch,False)
        tmp=model.get_predict(sess,feed_dict)
        predict.append(tmp)
    predict=np.concatenate(predict,axis=0)
    return predict

# 训练一个epoch 返回loss
def train_one_epoch(model,sess,train,batch_size):
    loss,task_loss=0.,0.
    # t0=time.time()
    np.random.shuffle(train)
    # t1=time.time()
    # print('shuffle train cost:{:.2f}s'.format(t1-t0))
    n_batch=train.shape[0]//batch_size+1
    for idx in range(n_batch):
        batch_data=get_batch_data(train,batch_size,idx)
        feed_dict=get_feed_dict(model,batch_data,True)

        _,batch_loss,batch_task_loss,batch_reg_loss=model.train(sess,feed_dict)
        loss+=batch_loss
        task_loss+=batch_task_loss

    loss/=n_batch
    task_loss/=n_batch

    return loss,task_loss

