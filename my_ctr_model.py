import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


class MyModel():

    # args是参数字典 
    def __init__(self,args,sparse_feature_dict,dense_feature_list,
                 text_feature_dict):
        
        self.lr=args.lr
        self.regs=eval(args.regs)
        self.emb_dim=args.emb_dim
        self.dnn_layer=eval(args.dnn_layer)

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
        
        # 线性特征参数
        weights_dict['liner']=tf.Variable(self.initializer([len(self.dense_feature_list),1]),
                                                  name='dense_w')

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

        # y
        input_dict['target']=tf.placeholder(tf.int32,shape=(None,),name='target')
        input_dict['dnn_keep_prob']=tf.placeholder(tf.float32,name='dnn_keep_prob')

        return input_dict
    
    # # 将input_dict中的所有变量做key 传真实values
    # def _get_feed_dict(self,values_dict):
    #     return dict(zip(self.input_dict.values(),values_dict.values()))
            

    # 构造模型
    def _forward(self):
        # 1 所有id查嵌入进行拼接
        self.embeddings=[]
        for id in self.sparse_feature_dict.keys():
            tmp=tf.nn.embedding_lookup(self.weights_dict[id],self.input_dict[id]) # [N,k]
            self.embeddings.append(tmp)

        self.embeddings=tf.concat(self.embeddings,axis=1) # [N,mK]

        # 2 liner侧
        self.liner_input=[tf.expand_dims(self.input_dict[tmp],axis=1) for tmp in self.dense_feature_list]
        self.liner_input=tf.concat(self.liner_input,axis=1) # [N,n]
        # self.liner_input=tf.expand_dims(tf.concat(self.liner_input,axis=1),axis=1) # [N,n]
        self.liner_ouput=tf.layers.dense(self.liner_input,1) # [N,1]
        
        # 3 传入NN

        # 3-1 : 构造dnn input,id类+liner
        self.concat_input=tf.concat([self.embeddings,self.liner_input],axis=1)
        # 3-2 : 构造dnn input, txt feature
         
        # 3-3 : 构造dnn input，是否对于id类显式两两建模
        pair_flag=True
        if(pair_flag):
            self.concat_input_emb=self.embeddings
            self.fields=tf.reshape(self.concat_input_emb,
                                   shape=[-1,len(self.sparse_feature_dict),self.emb_dim]) # [N,7,16]           

            # 写法二：将fm弄成[N,k]，concat，送入dnn
            self.sum_square=tf.reduce_sum(self.fields, axis=1,keepdims=False) #[N,16]
            self.sum_square=tf.square(self.sum_square) # [N,16]
            
            self.square_sum=tf.square(self.fields) # [N,7,16]
            self.square_sum=tf.reduce_sum(self.square_sum,axis=1,keepdims=False) # [N,16]
            
            self.second_order=0.5*tf.subtract(self.sum_square,self.square_sum) # [N,16]
            
            self.concat_input=tf.concat([self.concat_input,self.second_order],axis=1)

        # 3-4 多层dnn
        self.concat_output=self.concat_input
            
        for layer_size in self.dnn_layer:
            self.concat_output=tf.layers.dense(self.concat_output,layer_size,activation='relu')
            # self.concat_output=tf.layers.dense(self.concat_output,layer_size,
            #                                    activation='relu',kernel_initializer=self.initializer)
        
        # 最后一个dnn 加上dropout
        self.concat_output=tf.nn.dropout(self.concat_output, keep_prob=self.input_dict['dnn_keep_prob'])
            
        self.dnn_output=tf.layers.dense(self.concat_output,1) # [N,1]

        # 4 预测最后一层
        self.logit=self.dnn_output
        # self.logit=tf.add(self.dnn_output,self.liner_ouput) # [N,1]
            
        self.logit=tf.squeeze(self.logit,axis=1) # (N,)
        self.predict=tf.nn.sigmoid(self.logit)
        return self.predict


    # pair构造group的fm交互
    def _pair_fm(self,group,num):
        fields=tf.reshape(group,
                          shape=[-1,num,self.emb_dim]) # [N,3,16]
        sum_square=tf.reduce_sum(fields, axis=1,keepdims=False) #[N,16]
        sum_square=tf.square(sum_square) # [N,16]
            
        square_sum=tf.square(fields) # [N,3,16]
        square_sum=tf.reduce_sum(square_sum,axis=1,keepdims=False) # [N,16]
            
        second_order=0.5*tf.subtract(sum_square,square_sum) # [N,16]
            
        return second_order
      

    # 计算损失函数
    def _get_loss(self,predict,target):
        task_loss = tf.losses.log_loss(target, predict)
       
        reg_loss=tf.nn.l2_loss(self.embeddings)           
        #reg_loss=tf.nn.l2_loss(self.concat_input)+tf.nn.l2_loss(self.des_embedding)+tf.nn.l2_loss(self.asr_embedding)
        reg_loss=reg_loss*self.regs[0]
        return task_loss,reg_loss

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss,self.task_loss,self.reg_loss],feed_dict)

    # 得到预测
    def get_predict(self,sess,feed_dict):
        tmp=sess.run(self.predict,feed_dict)
        return tmp

