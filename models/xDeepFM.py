'''
- xDeepFM 2021/11/22 @Dive
'''
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


class xDeepFM():

    # args是参数字典 
    def __init__(self,args,sparse_feature_dict,dense_feature_list,
                 text_feature_dict):
        
        self.lr=args.lr
        self.regs=eval(args.regs)
        self.emb_dim=args.emb_dim
        self.dnn_layer=eval(args.dnn_layer)
        self.cin_layer=eval(args.cin_layer)

        self.sparse_feature_dict=sparse_feature_dict
        self.dense_feature_list=dense_feature_list

        self.field_num=len(self.sparse_feature_dict)
        
        # 1 初始化所有模型参数
        self.weights_dict=self._init_weights()
        # 1 定义输入placeholder
        self.input_dict=self._init_input()
        # 2 搭建模型
        self.predict=self._forward()
        # 3 计算损失函数
        self.task_loss,self.reg_loss=self._get_loss(self.predict,self.input_dict['target'])
        self.loss=self.task_loss+self.reg_loss
        # 4 优化器
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    # 初始化嵌入表和w
    def _init_weights(self):
        weights_dict=dict()
        self.initializer=tensorflow.contrib.layers.xavier_initializer()
        # 各个sparse feature的嵌入表 name:max_size
        for feature_name,size in self.sparse_feature_dict.items():
            weights_dict[feature_name]=tf.Variable(self.initializer([size,self.emb_dim]),
                                                  name='{}_embedding'.format(feature_name))
            weights_dict[feature_name+'_bias']=tf.Variable(self.initializer([size,self.emb_dim]),
                                                  name='{}_bias'.format(feature_name))
        
        # 线性特征参数
        weights_dict['dense_w']=tf.Variable(self.initializer([len(self.dense_feature_list),1]),
                                                  name='dense_w')

        # CIN 参数
        cin_w_list=[]
        for i in range(len(self.cin_layer)):
            pre_l=self.field_num if i==0 else self.cin_layer[i-1]
            cin_w_list.append(tf.Variable(self.initializer([self.cin_layer[i],pre_l,self.field_num]),
                                                  name='cin_w_{}_to_{}'.format(i,i+1)))
        weights_dict['cin_w_list']=cin_w_list

        return weights_dict

    # 定义模型输入placeholder
    def _init_input(self):
        input_dict=dict()
        # 分类特征 id类
        for feature_name in self.sparse_feature_dict.keys():
            input_dict[feature_name]=tf.placeholder(tf.int32,shape=(None,),name=feature_name)

        # 数值特征
        for feature_name in self.dense_feature_list:
            input_dict[feature_name]=tf.placeholder(tf.float32,shape=(None,),name=feature_name)

        # 序列型特征

        # y
        input_dict['target']=tf.placeholder(tf.int32,shape=(None,),name='target')
        input_dict['dnn_keep_prob']=tf.placeholder(tf.float32,name='dnn_keep_prob')

        return input_dict

    # 构造模型
    def _forward(self):
        # 1 所有id查嵌入进行拼接
        self.field_embeddings=[]
        for id in self.sparse_feature_dict.keys():
            tmp=tf.nn.embedding_lookup(self.weights_dict[id],self.input_dict[id]) # F_num个[N,k]
            self.field_embeddings.append(tmp)
        # 1-1 类别型id
        self.embeddings=tf.concat(self.field_embeddings,axis=1) # [N,F_num*k]
        self.field_embeddings=tf.stack(self.field_embeddings,axis=1) # [N,F_num,k]
        # 1-2 数值型id
        self.dense_input=[tf.expand_dims(self.input_dict[k],axis=1) for k in self.dense_feature_list]
        self.dense_input=tf.concat(self.dense_input,axis=1) # [N,Dense_num]

        # 2 liner侧
        self.liner_logit=self._linear_logit()

        # 3 CIN 层
        self.cin_output=self._cin_layer(self.field_embeddings)
        self.cin_logit=tf.layers.dense(self.cin_output,1,use_bias=False) # [N,1]
        
        # 4 NN侧
        self.dnn_input=tf.concat([self.embeddings,self.dense_input],axis=1)
        self.dnn_logit=self._dnn_logit(self.dnn_input)

        # 5 预测logit
        self.logit=self.liner_logit+self.cin_logit+self.dnn_logit # [N,1] 注:只有DNN加了bias 看作通用的bias
        self.logit=tf.squeeze(self.logit,axis=1) # (N,)
        self.predict=tf.nn.sigmoid(self.logit)

        return self.predict

    
    # 计算Linear侧logit
    def _linear_logit(self):
        sparse_bias=[]
        # 1 类别型field bias
        for id in self.sparse_feature_dict.keys():
            tmp=tf.nn.embedding_lookup(self.weights_dict['{}_bias'.format(id)],self.input_dict[id]) # F_num个[N,1]
            sparse_bias.append(tmp)
        sparse_bias=tf.concat(sparse_bias,axis=1) # [N,F_num]
        sparse_bias=tf.reduce_sum(sparse_bias,axis=1,keepdims=True) # [N,1]
        
        # 2 数值型field LR中的w
        dense_bias=tf.matmul(self.dense_input,self.weights_dict['dense_w']) # [N,D_num] [D_num,1] -> [N,1]

        return sparse_bias+dense_bias
    
    # 传入cin层
    def _cin_layer(self,X_0):
        '''
        X_0:[N,F_num,k]
        '''
        X_list=[]
        
        # 进行L层 输出L个X
        for i in range(len(self.cin_layer)):
            print('===============CIN:{} layer====================='.format(i+1))
            X_pre=X_0 if i==0 else X_list[-1] # [N,H_pre,k]
            w=self.weights_dict['cin_w_list'][i] # [H_cur,H_pre,H_0]

            X_cur=self._out_product(X_pre,X_0) # [N,H_pre,k] [N,H_0,k] -> [N,H_pre,H_0,k]
            X_cur=tf.expand_dims(X_cur,axis=4) # [N,H_pre,H_0,k,1]
            w=tf.expand_dims(w,axis=0) # [1,H_cur,H_pre,H_0]
            w=tf.expand_dims(w,axis=4) # [1,H_cur,H_pre,H_0,1]
            w=tf.transpose(w,perm=[0,2,3,4,1]) # [1,H_pre,H_0,1,H_cur]

            X_cur=X_cur*w # [N,H_pre,H_0,k,H_cur]
            print('max shape:{}'.format(X_cur.shape))
            X_cur=tf.reduce_sum(X_cur,axis=1,keepdims=False) # [N,H_0,k,H_cur]
            X_cur=tf.reduce_sum(X_cur,axis=1,keepdims=False) # [N,k,H_cur]
            X_cur=tf.transpose(X_cur,perm=[0,2,1]) # [N,H_cur,k]
            print('X shape:{}'.format(X_cur.shape))
            X_list.append(X_cur)
        
        print('======================================================')
        
        # 对每一层的 X 进行 pooling+concat
        output=[]

        for X in X_list:
            output.append(tf.reduce_sum(X,axis=2,keepdims=False)) # [N,H_cur,k] -> [N,H_cur]
        
        output=tf.concat(output,axis=1) # [N,H_1+H_2+H_3+..+H_L]
        return output
    
    def _out_product(self,A,B):
        # [N,a,k] [N,b,k] -> [N,a,b,k]
        dim1=A.get_shape().as_list()[1]
        dim2=B.get_shape().as_list()[1]
        output=[] # list

        # 得b个 [N,a,k]
        for i in range(dim2):
            output.append(A*B[:,i:i+1,:]) # [N,a,k] [N,1,k] -> [N,a,k]
        
        output=tf.stack(output,axis=2) # [N,a,b,k]
        return output

    # 计算DNN的logit
    def _dnn_logit(self,dnn_input):
        logit=dnn_input
        # 1 多层DNN
        for layer_size in self.dnn_layer:
            logit=tf.layers.dense(logit,layer_size,
                                    activation='relu',
                                    kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0))
        # 2 最后一层DNN+dropout
        logit=tf.nn.dropout(logit, keep_prob=self.input_dict['dnn_keep_prob'])
        # 3 加一个NN得logit
        logit=tf.layers.dense(logit,1,use_bias=True,
                                kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0)) # [N,1]
        return logit

    # 计算损失函数
    def _get_loss(self,predict,target):
        task_loss = tf.losses.log_loss(target, predict)
       
        reg_loss=tf.nn.l2_loss(self.embeddings)
        reg_w_loss=tf.nn.l2_loss(self.weights_dict['dense_w'])

        for k in self.sparse_feature_dict.keys():
            reg_w_loss=reg_w_loss+tf.nn.l2_loss(self.weights_dict[k+'_bias'])
        
        for w in self.weights_dict['cin_w_list']:
            reg_w_loss=reg_w_loss+tf.nn.l2_loss(w)
        
        # MLP中的w b
        reg_w_loss=reg_w_loss+tf.losses.get_regularization_loss()

        reg_loss=reg_loss*self.regs[0]+reg_w_loss*self.regs[1]
        return task_loss,reg_loss

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss,self.task_loss,self.reg_loss],feed_dict)

    # 得到预测
    def get_predict(self,sess,feed_dict):
        tmp=sess.run(self.predict,feed_dict)
        return tmp

