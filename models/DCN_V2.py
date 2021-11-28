'''
- DCN-V2 DCN-Mix 2021/11/26 @Dive
'''
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


class DCN_V2():

    # args是参数字典 
    def __init__(self,args,sparse_feature_dict,dense_feature_list,
                 text_feature_dict):
        
        self.lr=args.lr
        self.regs=eval(args.regs)
        self.emb_dim=args.emb_dim
        self.dnn_layer=eval(args.dnn_layer)
        self.cross_layer_num=args.cross_layer_num # cross layer num
        self.combine_type=args.combine_type # stack / parallel
        self.cross_type=args.cross_type # v1 / origin / rank / expert

        self.rank_dim=args.rank_dim # low rank mlp dim1 if
        self.rank2_dim=args.rank2_dim # low rank mlp dim2 if
        self.expert_num=args.expert_num # 专家网络数量

        self.sparse_feature_dict=sparse_feature_dict
        self.dense_feature_list=dense_feature_list
        
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
        '''
        input-output-logit
        '''
        # 1 所有id查嵌入进行拼接
        self.field_embeddings=[]
        for id in self.sparse_feature_dict.keys():
            tmp=tf.nn.embedding_lookup(self.weights_dict[id],self.input_dict[id]) # F_num个[N,k]
            self.field_embeddings.append(tmp)
        # 1-1 类别型id
        self.field_embeddings=tf.concat(self.field_embeddings,axis=1) # [N,F_num*k]
        # 1-2 数值型id 拼接起来
        self.dense_input=[self.input_dict[k] for k in self.dense_feature_list]
        self.dense_input=tf.stack(self.dense_input,axis=1) # [N,Dense_num]
        # 1-3 concat得到x_0
        self.field_embeddings=tf.concat([self.field_embeddings,self.dense_input],axis=1) # [N,F_num*k+D_num]

        print('=======================x_0 shape:{}========================'.format(self.field_embeddings.shape))

        # 2 选择结构
        self.output=None
        if(self.combine_type=='stack'):
            self.output=self.stack_cross(self.field_embeddings)
        elif(self.combine_type=='parallel'):
            self.output=self.parallel_cross(self.field_embeddings)
        
        # 3 预测logit
        self.logit=tf.layers.dense(self.output,1,use_bias=True,
                                    name='predict_layer') # [N,1]

        self.logit=tf.squeeze(self.logit,axis=1) # (N,)
        self.predict=tf.nn.sigmoid(self.logit)

        return self.predict
    
    # Stack 结构
    def stack_cross(self,input):
        # 1 Cross Layer
        output=self._multi_cross_layer(input)
        # 2 DNN Layer
        output=self._dnn_layer(output)
        return output
    
    # Parallel
    def parallel_cross(self,input):
        # 1 Cross Layer
        output=self._multi_cross_layer(input)
        # 2 DNN Layer
        dnn_output=self._dnn_layer(input)
        # 3 CONCAT
        output=tf.concat([output,dnn_output],axis=1)
        return output

    # 一层 Cross Layer V1
    def _cross_layer_v1(self,input,x_0,i=0):
        '''
        @input:[N,d] cur layer input = pre layer output
        @x_0:[N,d]
        @i:cur cross layer num
        return: cur cross layer output:[N,d]
        '''
        # x_{l} = x_{0} wise MLP(x_{l-1}) + x_{l-1}
        d=input.get_shape().as_list()[1]
        # 初始化v1 cross layer中的bias 因为input的dim不确定 所以在这里进行初始化
        if(i==0):
            self.weights_dict['v1_bias']=[]
            # Note:这里用j 别用i
            for j in range(self.cross_layer_num):
                self.weights_dict['v1_bias'].append(tf.Variable(self.initializer([1,d]),
                                                name='cross_{}_bias'.format(j)))

        output=tf.layers.dense(input,1,use_bias=False,activation=None,
                                kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0),
                                name='cross_layer_{}_mlp'.format(i)) # [N,d] -> [N,1]
        bias=self.weights_dict['v1_bias'][i] # [1,d]
        output=tf.multiply(x_0,output)+bias+input # [N,d] [N,1] + [1,d] + [N,d] -> [N,d]

        return output

    # 一层原始 Cross Layer V2
    def _cross_layer_v2(self,input,x_0,i=0):
        '''
        @input:[N,d] cur layer input = pre layer output
        @x_0:[N,d]
        @i:cur cross layer num
        return: cur cross layer output:[N,d]
        '''
        # x_{l} = x_{0} wise MLP(x_{l-1}) + x_{l-1}
        d=input.get_shape().as_list()[1]
        output=tf.layers.dense(input,d,use_bias=True,activation=None,
                                kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0),
                                name='cross_layer_{}_mlp'.format(i)) # [N,d] -> [N,d]
        output=tf.multiply(x_0,output)+input # [N,d] [N,d] +[N,d]

        return output

    # 一层低秩分解 Cross Layer
    def _cross_layer_low_rank(self,input,x_0,i=0):
        '''
        @input:[N,d] cur layer input = pre layer output
        @x_0:[N,d]
        @rank_dim: low rank dim
        @i:cur cross layer num
        return: cur cross layer output:[N,d]
        '''
        # x_{l} = x_{0} wise 2MLP(x_{l-1}) + x_{l-1}
        d=input.get_shape().as_list()[1]
        output=tf.layers.dense(input,self.rank_dim,use_bias=False,activation=None,
                                kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0),
                                name='cross_layer_{}_mlp1'.format(i)) # [N,d] -> [N,r]
        output=tf.layers.dense(output,d,use_bias=True,activation=None,
                                kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0),
                                name='cross_layer_{}_mlp2'.format(i)) # [N,r] -> [N,d]
        output=tf.multiply(x_0,output)+input # [N,d] [N,d] +[N,d]

        return output

    # 一层 DCN-Mix
    def _cross_layer_expert(self,input,x_0,i=0):
        '''
        @input:[N,d] cur layer input = pre layer output
        @x_0:[N,d]
        return: expert cross [N,d]
        '''
        d=input.get_shape().as_list()[1]
        output_list=[] # 每个expert的output
        # 1 Expert Network:计算K个子网络的输出
        for k in range(self.expert_num):
            # three mlp
            output=tf.layers.dense(input,self.rank_dim,use_bias=False,activation='relu',
                                    kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0),
                                    name='layer_{}_expert_{}_mlp1'.format(i,k)) # [N,d] -> [N,r]
            output=tf.layers.dense(output,self.rank2_dim,use_bias=False,activation='relu',
                                    kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0),
                                    name='layer_{}_expert_{}_mlp2'.format(i,k)) # [N,r] -> [N,r]
            output=tf.layers.dense(output,d,use_bias=True,activation=None,
                                    kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0),
                                    name='layer_{}_expert_{}_mlp3'.format(i,k)) # [N,r] -> [N,d]
            output=tf.multiply(x_0,output) # [N,d]
            output_list.append(output) # K个[N,d]
        
        # 2 Gate Network:根据input计算其在K个输出的权重
        gate=tf.layers.dense(input,self.expert_num,use_bias=True,activation=None,
                                kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0),
                                reuse=tf.AUTO_REUSE,
                                name='gate_network') # [N,d] -> [N,K]
        gate_sigmoid_flag=True
        gate=tf.nn.sigmoid(gate) if gate_sigmoid_flag else tf.nn.softmax(gate,axis=1) # [N,K]
        
        # 3 Sum:加权求和
        output=tf.stack(output_list,axis=2) # [N,d] -> [N,d,K]
        gate=tf.expand_dims(gate,axis=1) # [N,K] -> [N,1,K]
        output=output*gate # [N,d,K] [N,1,K] -> [N,d,K]
        output=tf.reduce_sum(output,axis=2) # [N,d,K] -> [N,d]
        output=output+input # [N,d]

        return output


    # 多层Cross Layer
    def _multi_cross_layer(self,input):
        '''
        @input:x_0 [N,d]
        return: multi cross output [N,d]
        '''
        d=input.get_shape().as_list()[1]
        x_pre=input # x_{l-1}
        output=None # x_{l}

        for i in range(self.cross_layer_num):
            # Cross Layer V1
            if(self.cross_type=='v1'):
                output=self._cross_layer_v1(x_pre,input,i=i)
            # Cross Layer V2
            elif(self.cross_type=='v2'):
                output=self._cross_layer_v2(x_pre,input,i=i)
            # Low Rank Cross Layer
            elif(self.cross_type=='rank'):
                output=self._cross_layer_low_rank(x_pre,input,i=i)
            # Expert Cross Layer
            elif(self.cross_type=='expert'):
                output=self._cross_layer_expert(x_pre,input,i=i)
            x_pre=output
        
        return output

    # 多层DNN
    def _dnn_layer(self,dnn_input):
        output=dnn_input
        # 1 多层DNN
        for layer_size in self.dnn_layer:
            output=tf.layers.dense(output,layer_size,
                                    activation='relu',
                                    kernel_regularizer=tensorflow.contrib.layers.l2_regularizer(scale=1.0))
        # 2 最后一层DNN+dropout
        output=tf.nn.dropout(output, keep_prob=self.input_dict['dnn_keep_prob'])
        return output

    # 计算损失函数
    def _get_loss(self,predict,target):
        task_loss = tf.losses.log_loss(target, predict)
       
        reg_loss=tf.nn.l2_loss(self.field_embeddings)
        reg_w_loss=tf.constant(0.,dtype=tf.float32)
        
        # 所有的w b
        reg_w_loss=reg_w_loss+tf.losses.get_regularization_loss()
        if(self.cross_type=='v1'):
            for b in self.weights_dict['v1_bias']:
                reg_w_loss=reg_w_loss+tf.nn.l2_loss(b)

        reg_loss=reg_loss*self.regs[0]+reg_w_loss*self.regs[1]
        return task_loss,reg_loss

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss,self.task_loss,self.reg_loss],feed_dict)

    # 得到预测
    def get_predict(self,sess,feed_dict):
        tmp=sess.run(self.predict,feed_dict)
        return tmp

