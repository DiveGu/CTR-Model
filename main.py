'''
- main 2021/11/22 @Dive
- 训练模型入口
'''
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow
import time
import datetime
import argparse
import re
import os
import json

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import multiprocessing
cores = multiprocessing.cpu_count()
# print('cpu:{}'.format(cores))

from data_make import make_dataset
from models.xDeepFM import xDeepFM
from models.FiBiNET import FiBiNET
from models.DCN_V2 import DCN_V2
from models.FFM import FFM
from utils import train_one_epoch,predict_by_batch


SEED=2021
tf.set_random_seed(SEED)

# 存储数据的根目录
root_path='/testcbd017_gujinfang/GJFCode/ccf_predict'


# 设置模型参数
def get_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='FFM')
    
    parser.add_argument('--emb_dim',type=int,default=4)
    parser.add_argument('--regs', nargs='?',default='[1e-5,1e-5,1e-3]')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dnn_keep', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_args()

# 从train中划分训练集和验证集
def make_train_val(df):
    train, val = train_test_split(df, test_size=0.25)
    return train,val

def main():
    args=get_args()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 1 输入数据 train val test
    train,test,sparse_feats_dict,dense_feats=make_dataset()
    train,val=make_train_val(train)

    print('train num:{},val num:{},test num:{}'.format(
        train.shape,val.shape,test.shape
    ))

    # 2 sparse和dense类型配置
    dense_feats_list=dense_feats

    print('use cols num:{},sp num:{},dense num:{}'.format(
        len(sparse_feats_dict)+len(dense_feats_list),
        len(sparse_feats_dict),len(dense_feats_list),
    ))

    # 3 构造模型
    # model=FiBiNET(args,sparse_feats_dict,dense_feats_list,dict({}))
    # model=xDeepFM(args,sparse_feats_dict,dense_feats_list,dict({}))
    # model=DCN_V2(args,sparse_feats_dict,dense_feats_list,dict({}))

    from collections import defaultdict
    field_2_feat=defaultdict(list) # field id:[feature name]
    feat_name=list(sparse_feats_dict.keys())+dense_feats_list
    # 示例：按照feature idx划分field 并不是真正的对应关系
    for k in range(6):
        start_idx=len(feat_name)//6*k
        end_idx=min(len(feat_name)//6*(k+1),len(feat_name))
        for idx in range(start_idx,end_idx):
            field_2_feat[k].append(feat_name[idx])

    print(field_2_feat)
    model=FFM(args,sparse_feats_dict,dense_feats_list,dict({}),field_2_feat)
    
    
    
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
 
    # 4 训练epoch
    best_auc=0.0

    for epoch in range(args.epochs):
        # 4-1 训练一个epochj
        t0=time.time()
        loss,task_loss=train_one_epoch(model,sess,train,args.batch_size)
        t1=time.time()
        print('************************************************************************')
        print('epoch:{},loss:{:.5f},task_loss:{:.5f}'.format(epoch+1,loss,task_loss))

        # 4-2 在验证集上计算auc

        train_predict_ans=predict_by_batch(model,sess,train,args.batch_size*4)
        train_auc=roc_auc_score(train[:,-1],train_predict_ans)

        val_predict_ans=predict_by_batch(model,sess,val,args.batch_size*4)
        auc=roc_auc_score(val[:,-1],val_predict_ans)
        print('train auc:{},val auc:{},train cost:{:.2f}s'.format(train_auc,auc,t1-t0))

        # 4-3 更新当前day当前模型的最好auc 
        if(auc>=best_auc):
            best_auc=auc


    # # 保存sub_dict


if __name__=='__main__':
    main()
            
