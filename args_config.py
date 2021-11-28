'''
- args_config 2021/11/22 @Dive
- 各种参数配置
'''

# 设置模型参数
def get_args():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='mybasemodel')

    parser.add_argument('--emb_dim',type=int,default=16)
    parser.add_argument('--dnn_layer',nargs='?',default='[128,128,64]')
    parser.add_argument('--regs', nargs='?',default='[1e-5,1e-6,1e-3]')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dnn_keep', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)

    return parser.parse_args()

# xDeepFM 参数
def get_args_xDeepFM():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='xDeepFM')
    
    parser.add_argument('--emb_dim',type=int,default=16)
    parser.add_argument('--dnn_layer',nargs='?',default='[128,128,64]')
    parser.add_argument('--regs', nargs='?',default='[1e-6,1e-5,1e-3]')
    parser.add_argument('--cin_layer',nargs='?',default='[16,10]')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dnn_keep', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_args()

# FiBiNET 参数
def get_args_FiBiNET():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='FiBiNET')
    
    parser.add_argument('--emb_dim',type=int,default=4)
    parser.add_argument('--dnn_layer',nargs='?',default='[1024,1024,512]')
    parser.add_argument('--regs', nargs='?',default='[1e-6,1e-5,1e-3]')
    parser.add_argument('--dim_rate',type=int,default=2)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dnn_keep', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_args()

# DCN_V2 参数
def get_args_DCN_V2():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='DCN_V2')
    
    parser.add_argument('--emb_dim',type=int,default=8)
    parser.add_argument('--dnn_layer',nargs='?',default='[256,256,128]')
    parser.add_argument('--regs', nargs='?',default='[1e-6,1e-5,1e-3]')
    parser.add_argument('--cross_type',nargs='?',default='rank') # v1 / v2 / rank / expert
    parser.add_argument('--combine_type',nargs='?',default='parallel') # stack / parallel

    parser.add_argument('--cross_layer_num',type=int,default=3)
    parser.add_argument('--expert_num',type=int,default=3)
    parser.add_argument('--rank_dim',type=int,default=128)
    parser.add_argument('--rank2_dim',type=int,default=128)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dnn_keep', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_args()


# FFM 参数
def get_args_xDeepFM():
    parser = argparse.ArgumentParser(description="Run MyModel.")
    # 模型参数
    parser.add_argument('--model_name',nargs='?',default='FFM')
    
    parser.add_argument('--emb_dim',type=int,default=16)
    parser.add_argument('--dnn_layer',nargs='?',default='[128,128,64]')
    parser.add_argument('--regs', nargs='?',default='[1e-6,1e-5,1e-3]')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dnn_keep', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_args()



