'''
- FFM 2021/11/27 @Dive
- Note：代码里的field指的是某一类feature
        high-field指的是FFM中的field 可以看做某几类features的集合
'''
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow


class FFM():

    # args是参数字典 
    def __init__(self,args,sparse_feature_dict,dense_feature_list,
                 text_feature_dict,field_2_feat):
        self.lr=args.lr
        self.regs=eval(args.regs)
        self.emb_dim=args.emb_dim

        self.sparse_feature_dict=sparse_feature_dict
        self.dense_feature_list=dense_feature_list
        self.field_2_feat=field_2_feat
        self.high_field_num=len(self.field_2_feat) # high field num
        
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
        # sparse feature feat_name:max_size
        # field_2_feat field_id:[feat_name]
        for i in range(self.high_field_num):
            for j in range(self.high_field_num):
                for feat in self.field_2_feat[i]:
                    name='field{}to{}_{}_embedding'.format(i,j,feat)
                    weights_dict[name]=self._init_embedding(i,j,feat)

        return weights_dict

    # 初始化 <field i,field j> 交互时feat的嵌入
    def _init_embedding(self,i,j,feat):
        if(feat in self.sparse_feature_dict):
            return tf.Variable(self.initializer([self.sparse_feature_dict[feat],self.emb_dim]),
                                name='field{}to{}_{}_embedding'.format(i,j,feat))
        else:
            return tf.Variable(self.initializer([1,self.emb_dim]),
                                name='field{}to{}_{}_embedding'.format(i,j,feat))

    # 获取 <field i,field j> 交互时feat的嵌入
    def _lookup_embedding(self,i,j,feat):
        name='field{}to{}_{}_embedding'.format(i,j,feat)
        # 1 分类型特征
        if(feat in self.sparse_feature_dict):
            return tf.nn.embedding_lookup(self.weights_dict[name],
                                        self.input_dict[feat]) # [N,k]
        # 2 连续型特征
        else:
            output=tf.expand_dims(self.input_dict[feat],axis=1) # (N,) -> [N,1]
            output=tf.matmul(output,self.weights_dict[name]) # [N,k]
            # print(output.shape)
            return output

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
        # 1 按照field顺序得特征嵌入
        self.field_embeddings=[[None]*self.high_field_num for _ in range(self.high_field_num)]
        # field[i][j][f] 表示field i中的第f个feature对于field j使用的嵌入 
        for i in range(self.high_field_num):
            for j in range(self.high_field_num):
                self.field_embeddings[i][j]=[]
                # field i中的所有feature
                for feat in self.field_2_feat[i]:
                    self.field_embeddings[i][j].append(self._lookup_embedding(i,j,feat))
        
        for i in range(self.high_field_num):
            for j in range(self.high_field_num):
                # print(self.field_embeddings[i][j][0].shape)
                self.field_embeddings[i][j]=tf.stack(self.field_embeddings[i][j],axis=1) # [N,F,k]

        # 2 计算二阶交互
        flag=False
        for i in range(self.high_field_num):
            # <field i,field i>
            if(not flag):
                second_order=self._fm_logit(self.field_embeddings[i][i])
                flag=True
            else:
                second_order+=self._fm_logit(self.field_embeddings[i][i])
            # <field i,field j>
            for j in range(i+1,self.high_field_num):
                second_order+=self._second_order(self.field_embeddings[i][j],self.field_embeddings[j][i])
        
        self.logit=second_order # [N,1]
        self.predict=tf.nn.sigmoid(tf.squeeze(self.logit,axis=1)) # (N,)

        return self.predict

    # 获取两个field的二阶交互项
    def _second_order(self,e1,e2):
        '''
        Suppose 3 features in field i;4 features in field j
        @e1:field i embeddings [N,3,k]
        @e2:field j embeddings [N,4,k]
        '''
        output=[]
        d1=e1.get_shape().as_list()[1]
        for i in range(d1):
            tmp=e1[:,i:i+1,:]*e2 # [N,1,k] [N,4,k] -> [N,4,k]
            tmp=tf.reduce_sum(tmp,axis=[1,2],keepdims=False) # [N,4,k] -> (N,)
            output.append(tmp)
        
        output=tf.stack(output,axis=1) # (N,) -> [N,3]
        output=tf.reduce_sum(output,axis=1,keepdims=True) # [N,1]
        return output

    # 获取fm项
    def _fm_logit(self,input):
        '''
        @input:[N,F,k]
        return: FM logit of input [N,1]
        '''
        # print(input.shape)
        sum_inten=tf.reduce_sum(input,axis=1,keepdims=False) # [N,F,k] -> [N,k]
        sum_inten=sum_inten*sum_inten # [N,k]
        # print(sum_inten.shape)
        sum_inten=tf.reduce_sum(sum_inten,axis=1,keepdims=True) # [N,k] -> [N,1]

        self_inten=input*input # [N,F,k] [N,F,k] -> [N,F,k]
        self_inten=tf.reduce_sum(self_inten,axis=2,keepdims=False) # [N,F,k] -> [N,F]
        self_inten=tf.reduce_sum(self_inten,axis=1,keepdims=True) # [N,F] -> [N,1]

        return 0.5*(sum_inten-self_inten)


    # 计算损失函数
    def _get_loss(self,predict,target):
        task_loss = tf.losses.log_loss(target, predict)
        
        reg_loss=tf.nn.l2_loss(self.field_embeddings)
        reg_loss=reg_loss*self.regs[0]
        
        return task_loss,reg_loss

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss,self.task_loss,self.reg_loss],feed_dict)

    # 得到预测
    def get_predict(self,sess,feed_dict):
        tmp=sess.run(self.predict,feed_dict)
        return tmp

