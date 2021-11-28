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

from my_ctr_model import MyModel

SEED=2021
tf.set_random_seed(SEED)

# 存储数据的根目录
root_path='/testcbd017_gujinfang/GJFCode/ccf_predict'


# 1 设置模型参数
# 2 生成feed需要的train val test
#   - 读取最原始raw data
#   - 数据基本的预处理 类别str转id、uid进行transform
#   - 特征工程
#   - 分清楚sparse、dense特征
#   - 对sparse特征进行fillna,对dense特征进行标准化
# 3 写model_train函数
# 4 写model_predict函数



# 设置模型参数
def get_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='mybasemodel')
    
    parser.add_argument('--emb_dim',type=int,default=16)
    parser.add_argument('--dnn_layer',nargs='?',default='[128,128,64]')
    parser.add_argument('--regs', nargs='?',default='[1e-6,1e-6,1e-3]')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dnn_keep', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5)

    return parser.parse_args()

def get_df_original(train_concat_flag):
    train_bank = pd.read_csv(root_path+'/train_public.csv')
    train_internet = pd.read_csv(root_path+'/train_internet.csv')
    train_bank=train_bank.rename(columns={"isDefault": "is_default"})
    test = pd.read_csv(root_path+'/test_public.csv')
    
    common_cols = []
    for col in train_bank.columns:
        if col in train_internet.columns:
            common_cols.append(col)      
    
    if(train_concat_flag):
        train_data = pd.concat([train_internet[common_cols], train_bank[common_cols]]).reset_index(drop=True)
    else:
        train_data=train_internet[common_cols]
        
    test_data = test[common_cols[:-1]]
    
    return train_data,test_data

def clean_mon(x):
    mons = {'jan':1, 'feb':2, 'mar':3, 'apr':4,  'may':5,  'jun':6,
            'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    year_group = re.search('(\d+)', x)
    if year_group:
        year = int(year_group.group(1))
        if year < 22:
            year += 2000
        elif 100 > year > 22:
            year += 1900
        else:
            year = 2022
    else:
        year = 2022

    month_group = re.search('([a-zA-Z]+)', x)
    if month_group:
        mon = month_group.group(1).lower()
        month = mons[mon]
    else:
        month = 0

    return year*100 + month
    
def df_process(df):
    # 1 处理时间
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    # 提取多尺度特征
    df['issue_date_y'] = df['issue_date'].dt.year
    df['issue_date_m'] = df['issue_date'].dt.month
    # 提取时间diff
    # 设置初始的时间
    base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    # 转换为天为单位
    df['issue_date_diff'] = df['issue_date'].apply(lambda x: x-base_time).dt.days
    #print(df[['issue_date', 'issue_date_y', 'issue_date_m', 'issue_date_diff']].head())
    df.drop('issue_date', axis = 1, inplace = True)
    # 2 类型特征 转为数字
    cols=['employer_type','industry',]
    for col in cols:
        #print(df[col].dtypes)
        lbe = LabelEncoder()
        lbe.fit(df[col].astype(str))
        df[col] = lbe.fit_transform(df[col].astype(str))
    # 3 处理特殊列 map 数字替换
    work_year_map={
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,'5 years': 5, 
        '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,'10+ years': 10,}

    df['work_year'] = df['work_year'].map(work_year_map)
    df['work_year'].fillna(-1, inplace=True)

    df['class'] = df['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

    # earlies_credit_mon 格式 'Sep-2005'
    df['earlies_credit_mon']=df['earlies_credit_mon'].apply(lambda x:clean_mon(x))

    return df

# 1 频率编码
# df[col] 中数量占比
def freq_enc(df, col):
    vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
    df[f'{col}_freq'] = df[col].map(vc)
    return df

## 2 目标编码
def stat(df, df_merge, group_by, agg):
    # 按照df分组计算agg
    group = df.groupby(group_by).agg(agg)
    # print(type(group))
    # print(group)

    columns = []
    for on, methods in agg.items():
        for method in methods:
            columns.append('{}_{}_{}'.format('_'.join(group_by), on, method))
    group.columns = columns
    group.reset_index(inplace=True)
    # 在df_merge中拼接df算出来的agg
    df_merge = df_merge.merge(group, on=group_by, how='left')

    del (group)
    gc.collect()
    return df_merge

def statis_feat(df_know, df_unknow):
    # df_unknow = stat(df_know, df_unknow, ['total_loan'], {'is_default': ['mean','median']})
    # df_unknow = stat(df_know, df_unknow, ['interest'], {'is_default': ['mean','median']})
    df_unknow = stat(df_know, df_unknow, ['work_year'], {'is_default': ['mean','std']})
    df_unknow = stat(df_know, df_unknow, ['class'], {'is_default': ['mean','std']})
    df_unknow = stat(df_know, df_unknow, ['employer_type'], {'is_default': ['mean','std']})
    df_unknow = stat(df_know, df_unknow, ['region'], {'is_default': ['mean','std']})
    
    return df_unknow  

## 3 统计特征 
# 各种分类特征下dense数据的统计
def brute_force(df, features, groups):
    for method in ['mean']:
        for feature in features:
            for group in groups:
                df[f'{group}_{feature}_{method}'] = df.groupby(group)[feature].transform(method)

    return df

def df_feat_make(df):
    # print(df.dtypes)
    # 特征feat所占的频率
    for feat in ['house_exist','issue_date_y','issue_date_m','region']:
        df = freq_enc(df, feat)
        
    df=statis_feat(df[df['is_default'].notna()], df)
    
    dense_feats=['total_loan', 'interest','debt_loan_ratio',]
    cate_feats=['class','work_year','employer_type','issue_date_m']
    df=brute_force(df, dense_feats, cate_feats)
    # 交叉特征
    df['total_loan_add_ratio'] = df['total_loan'] * df['debt_loan_ratio'] * df['year_of_loan']
    df['score_add']=df['scoring_high'] - df['scoring_low']
    
    return df

def make_feat_cols(df,dense_cols):
    # 【1】确定离散和连续列
    sparse_feats=list(set((df.select_dtypes(include='int').columns))-set(dense_cols))
    dense_feats=list(set(df.columns)-set(sparse_feats)-set(['is_default']))
    
    # 例如uid这种需要transform的 按照数量重新编码
    for col in ['loan_id','user_id','earlies_credit_mon','issue_date_y',]:
        lbe = LabelEncoder()
        lbe.fit(df[col].astype(str))
        df[col] = lbe.fit_transform(df[col].astype(str))
                                                   
    # 【2】按照列类型进行fill na 归一化
    for col in sparse_feats:
        df[col]=df[col].fillna(value=-1)
        df[col]=df[col]+1
        
    for col in dense_feats:
        if(col in ['is_default']):
            continue
        df[col]=df[col].fillna(value=df[col].mean())
        df[col]=StandardScaler().fit_transform(df[col].values.reshape(-1,1))
                                                        
    return df,sparse_feats,dense_feats

# 制作数据集
def make_dataset():
    t0=time.time()
    # 【1】读取原始数据
    train_data,test_data=get_df_original(True)
    df_all=pd.concat([train_data, test_data])
    # df_all=df_all.sample(frac=0.1)
    # 【2】预处理
    df_all=df_process(df_all)
    # 【3】特征工程
    # df_all=df_feat_make(df_all)
    # 【4】划分sparse、dense列
    df_all,sparse_feats,dense_feats=make_feat_cols(df_all,['issue_date_diff','work_year'])

    # 【5】拆分train test 保存
    # train = df_all[df_all['is_default'].notna()]
    # test  = df_all[df_all['is_default'].isna()]
    t1=time.time()
    print('make dataset cost:{:.2f}s'.format(t1-t0))
    return df_all,sparse_feats,dense_feats

# 从train中划分训练集和验证集
def make_train_val(df):
    train, val = train_test_split(df, test_size=0.25)
    return train,val

# 加载训练数据 测试数据
def get_batch_data(df,batch_size,idx):
    start=idx*batch_size
    end=(idx+1)*batch_size
    end=end if end<=df.shape[0] else df.shape[0]
    return df[start:end]

# 把df转为feed_dict
def convert_feed_dict(df,cols,y):
    feed_dict=dict()
    for c in cols:
        feed_dict[c]=df[c].values         
    feed_dict['dnn_keep_prob']=1.0

    # label
    if(y in df.columns):
        feed_dict['target']=df[y].values
        feed_dict['dnn_keep_prob']=1.0

    return feed_dict

# 改成并行
def get_batch_feed_dict(df,batch_size,cols,y,idx):
    tmp=get_batch_data(df,batch_size,idx)
    return convert_feed_dict(tmp,cols,y)
    

def convert_feed_list(df,cols,batch_size,y):
    batch_list=[]
    df=df[cols+[y]]
    n_batch=len(df)//batch_size+1
    # pool = multiprocessing.Pool(cores)
    
    # params=[(df,batch_size,cols,y,idx) for idx in range(0,n_batch)]
    # batch_list=pool.map(get_batch_feed_dict,params)
    
    for idx in range(n_batch):
        start=idx*batch_size
        end=(idx+1)*batch_size
        end=end if end<=df.shape[0] else df.shape[0]
        batch_df=df[start:end]
        
        feed_dict=dict()
        for c in cols:
            feed_dict[c]=batch_df[c].values
            
        feed_dict['dnn_keep_prob']=1.0
        if(c in df.columns):
            feed_dict['target']=batch_df[y].values
            feed_dict['dnn_keep_prob']=1.0
        
        batch_list.append(feed_dict)

    return batch_list


# 按照batch_size对df进行预测
def predict_by_batch(model,sess,df,batch_size,cols,y):
    predict=[]
    n_batch=df.shape[0]//batch_size+1
    for idx in range(n_batch):
        batch=get_batch_data(df,batch_size,idx)
        values_dict=convert_feed_dict(batch,cols,y)
        feed_dict=model._get_feed_dict(values_dict)
        tmp=model.get_predict(sess,feed_dict)
        predict.append(tmp)
    predict=np.concatenate(predict,axis=0)
    return predict

def main():
    args=get_args()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 1 输入数据 train val test
    df_all,sparse_feats,dense_feats=make_dataset()
    train = df_all[df_all['is_default'].notna()]
    test  = df_all[df_all['is_default'].isna()]
    train,val=make_train_val(train)

    print('train num:{},val num:{},test num:{}'.format(
        train.shape[0],val.shape[0],test.shape[0]
    ))

    # 2 sparse和dense类型配置
    sparse_feature_dict=dict()
    dense_feature_list=dense_feats

    for k in sparse_feats:
        if(k not in 'loan_id'):
            sparse_feature_dict[k]=df_all[k].max()+1

    print('use cols num:{},sp num:{},dense num:{}'.format(
        len(sparse_feature_dict)+len(dense_feature_list),
        len(sparse_feature_dict),len(dense_feature_list),
    ))

    # 3 构造模型
    model=MyModel(args,sparse_feature_dict,dense_feature_list,dict({}))
    
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # t0=time.time()
    # batch_list=convert_feed_list(train,sparse_features+dense_features,args.batch_size)
    # t1=time.time()
    # print('creat batch list cost:{:.2f}s'.format(t1-t0))
    
    # 4 训练epoch
    all_feat_cols=list(sparse_feature_dict.keys())+dense_feature_list
    best_auc=0.0

    for epoch in range(args.epochs):
        t0=time.time()
        loss,task_loss=0.,0.
        train=train.sample(frac=1.0)
        n_batch=train.shape[0]//args.batch_size+1
        for idx in range(n_batch):
            # values_dict=batch_list[idx]
            batch_data=get_batch_data(train,args.batch_size,idx)
            values_dict=convert_feed_dict(batch_data,all_feat_cols,'is_default')

            feed_dict=model._get_feed_dict(values_dict)
            _,batch_loss,batch_task_loss,batch_reg_loss=model.train(sess,feed_dict)
            loss+=batch_loss
            task_loss+=batch_task_loss

        loss/=n_batch
        task_loss/=n_batch
        print('************************************************************************')
        print('epoch:{},loss:{:.5f},task_loss:{:.5f}'.format(epoch+1,loss,task_loss))
        t1=time.time()

        # 4-1 在验证集上计算auc
        val_predict_ans=predict_by_batch(model,sess,val,args.batch_size*4,
                                        all_feat_cols,'is_default')

        t2=time.time()
        auc=roc_auc_score(val['is_default'].values,val_predict_ans)
        t3=time.time()
        print('val auc:{},train cost:{:.2f}s,auc cost:{:.2f}s'.format(auc,t1-t0,t3-t2))

        # 4-2 更新当前day当前模型的最好auc 
        if(auc>=best_auc):
            best_auc=auc


    # 保存sub_dict


if __name__=='__main__':
    main()
            
