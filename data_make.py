'''
- data_make 2021/11/22 @Dive
- 准备模型的输入 包括原始数据读取、预处理、特征工程、sparse和dense特征列处理
'''
import numpy as np
import pandas as pd
import time
import datetime
import re
import os
import gc
import sys

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,KBinsDiscretizer


# 存储数据的根目录
root_path='/testcbd017_gujinfang/GJFCode/ccf_predict'

# *********************1 读取原始数据*****************************************
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

# *********************2 原始数据预处理*****************************************   
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

# *********************3 特征工程*****************************************
# 3-1 频率编码
# df[col] 中数量占比
def freq_enc(df, col):
    vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
    df[f'{col}_freq'] = df[col].map(vc)
    return df

## 3-2 目标编码
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

## 3-3 统计特征 
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

# 新增分桶列
def df_make_bin(df,col_list,k,type='uniform'):
    '''
    @type:分桶方式 quantile等频 uniform等距
    '''
    # 分桶KBinsDiscretizer quantile/uniform
    k_bin=KBinsDiscretizer(n_bins=[k],encode='ordinal',strategy=type)
    for col in col_list:  
        df[col+'_bin']=df[col]
        df[col+'_bin']=df[col+'_bin'].fillna(value=df[col].min())
        x=df[col+'_bin'].values.reshape(-1,1)
        # print(x)
        df[col+'_bin']=k_bin.fit(x).transform(x)
        # print(df[col+'_bin'].values)
        # sys.exit()
        df[[col+'_bin']]=df[[col+'_bin']].astype(int)
    
    return df


# 确定特征列所属类型 预处理
def process_feat_cols(df,no_sparse_cols,no_dense_cols):
    # 【1】确定离散和连续列
    int_cols=df.select_dtypes(include='int').columns
    # sparse特征列名、dense特征列名
    sparse_feats=list(set(int_cols)-set(no_sparse_cols))
    dense_feats=list(set(df.columns)-set(sparse_feats)-set(no_dense_cols)-set(['is_default']))
    # 类别特征列名：类别数量
    sparse_feats_dict=dict()
    
    # 例如uid这种需要transform的 按照数量重新编码
    for col in ['user_id','earlies_credit_mon','issue_date_y',]:
        lbe = LabelEncoder()
        lbe.fit(df[col].astype(str))
        df[col] = lbe.fit_transform(df[col].astype(str))
                                                   
    # 【2-1】sparse类型进行fill na 归一化
    for col in sparse_feats:
        df[col]=df[col].fillna(value=-1)
        df[col]=df[col]+1
        sparse_feats_dict[col]=df[col].max()+1
        
    # 【2-2】dense类型进行fill na 归一化
    for col in dense_feats:
        df[col]=df[col].fillna(value=0.)
        df[col]=StandardScaler().fit_transform(df[col].values.reshape(-1,1))

    # 【3】对于数值型列 分桶转化为类型列
    bin_flag=True
    bin_k=10
    bin_cols=list(set(dense_feats)-set(['is_default']))[:1]
    if(bin_flag):
        df=df_make_bin(df,bin_cols,bin_k,type='uniform') # 新增 col_bin列
        for col in bin_cols:
            df[col+'_bin']=df[col+'_bin']+1
            sparse_feats_dict[col+'_bin']=bin_k+1 # 将新增列添加到sparse_dict中

                                                       
    return df,sparse_feats_dict,dense_feats

# *********************4 制作数据集*****************************************
def make_dataset():
    t0=time.time()
    # 【1】读取原始数据
    train_data,test_data=get_df_original(True)
    df_all=pd.concat([train_data, test_data])
    # df_all=df_all.sample(frac=0.1)
    # 【2】预处理
    df_all=df_process(df_all)
    # 【3】特征工程
    df_all=df_feat_make(df_all)
    # 【4】划分sparse、dense列
    df_all,sparse_feats_dict,dense_feats=process_feat_cols(df_all,
                                                            ['user_id','issue_date_diff','work_year','loan_id'],
                                                            ['user_id','loan_id'])

    # 【5】拆分train test 保存
    train = df_all[df_all['is_default'].notna()]
    test  = df_all[df_all['is_default'].isna()]

    train=train[list(sparse_feats_dict.keys())+dense_feats+['is_default']].values
    test=test[list(sparse_feats_dict.keys())+dense_feats].values

    t1=time.time()
    print('make dataset cost:{:.2f}s'.format(t1-t0))
    # print(train[1,:])

    return train,test,sparse_feats_dict,dense_feats




