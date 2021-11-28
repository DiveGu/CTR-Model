'''
- FiBiNET 2021/11/24 @Dive
'''
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


class FiBiNET():

    # args是参数字典 
    def __init__(self,args,sparse_feature_dict,dense_feature_list,
                 text_feature_dict):
        
        self.lr=args.lr
        self.regs=eval(args.regs)
        self.emb_dim=args.emb_dim
        self.dnn_layer=eval(args.dnn_layer)
        self.dim_rate=args.dim_rate # W1 [F,F/r]

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
        # SENET两个MLP的参数 W1 W2
        weights_dict['S_W_list']=[]
        weights_dict['S_W_list'].append(tf.Variable(self.initializer([self.field_num,int(self.field_num/self.dim_rate)]),
                                        name='SENET_W1'))
        weights_dict['S_W_list'].append(tf.Variable(self.initializer([int(self.field_num/self.dim_rate),self.field_num]),
                                        name='SENET_W2'))
        # 双线性特征交互 参数 W
        for i in range(self.field_num):
            for j in range(i+1,self.field_num):
                weights_dict['W_{}_to_{}'.format(i,j)]=tf.Variable( 
                                                            self.initializer([self.emb_dim,self.emb_dim]),
                                                            name='W_{}_to_{}'.format(i,j))

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
        self.field_embeddings=tf.stack(self.field_embeddings,axis=1) # [N,F_num,k]
        # 1-2 数值型id 拼接起来
        self.dense_input=[self.input_dict[k] for k in self.dense_feature_list]
        self.dense_input=tf.stack(self.dense_input,axis=1) # [N,Dense_num]

        # 2 SENET层
        self.field_embeddings_senet=self._SENET_layer(self.field_embeddings,self.weights_dict['S_W_list']) # [N,F,k]

        # 3 双线性交互层
        self.original_binet=self._Bilinear_interaction(self.field_embeddings) # [N,F*(F-1)*l]
        self.senet_binet=self._Bilinear_interaction(self.field_embeddings_senet) # [N,F*(F-1)*l]
        
        # 4 CONCAT+DNN层
        self.dnn_input=tf.concat([self.original_binet,self.senet_binet,self.dense_input],axis=1) # [N,2*F(F-1)*k+D_num]
        self.dnn_logit=self._dnn_logit(self.dnn_input) # [N,1]
        print('======================DNN input shape:{}======================='.format(self.dnn_input.shape))
        
        # 5 预测logit
        self.logit=tf.squeeze(self.dnn_logit,axis=1) # (N,)
        self.predict=tf.nn.sigmoid(self.logit)

        return self.predict

    # SENET Layer
    def _SENET_layer(self,input,W_list):
        '''
        @input:[N,F,k]
        @W_list:FC的参数W
        return: re-weight input
        '''
        # Squeeze
        output=tf.reduce_mean(input,axis=2,keepdims=False) # [N,F,k] -> [N,F]
        # Excitation
        output=tf.nn.tanh(tf.matmul(output,W_list[0])) # [N,F] -> [N,F/r]
        output=tf.nn.sigmoid(tf.matmul(output,W_list[1])) # [N,F/r] -> [N,F]
        # Re-Weight
        output=input*tf.expand_dims(output,axis=2) # [N,F,k] [N,F,1] -> [N,F,k]

        return output

    # Bilinear Interaction Layer
    def _Bilinear_interaction(self,input):
        '''
        @input: [N,F,k]
        return: [N,F(F-1)/2]
        '''
        output=[] # F(F-1)个[N,k]

        for i in range(self.field_num):
            for j in range(i+1,self.field_num):
                w=self.weights_dict['W_{}_to_{}'.format(i,j)]
                output.append(tf.multiply(tf.matmul(input[:,i,:],w),input[:,j,:])) # [N,k] [k,k] [N,k] -> [N,k]
        
        output=tf.concat(output,axis=1) # [N,F(F-1)*k]

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
       
        reg_loss=tf.nn.l2_loss(self.field_embeddings)
        reg_w_loss=tf.constant(0.,dtype=tf.float32)

        for w in self.weights_dict['S_W_list']:
            reg_w_loss=reg_w_loss+tf.nn.l2_loss(w)
        
        for i in range(self.field_num):
            for j in range(i+1,self.field_num):
                reg_w_loss=reg_w_loss+tf.nn.l2_loss(self.weights_dict['W_{}_to_{}'.format(i,j)])
        
        # DNN中的w b
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

