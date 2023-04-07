import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils  import data 
import  argparse
import matplotlib.pyplot as plt

'''
优化方向：
    dropout、增加一层神经网络
'''

'''定义参数'''
def parse_args():
    parse=argparse.ArgumentParser(description="神经网络超参数")  #创建参数对象
    parse.add_argument("--lr",default=7,type=float,help="学习率")
    parse.add_argument("--w_decay",default=0.01,type=float,help="权重衰退")
    parse.add_argument("--num_epochs",default=100,type=int,help="训练次数")
    parse.add_argument("--batch_size",default=64,type=int)
    parse.add_argument("--k",default=5,type=int,help="定义k折交叉验证")
    parse.add_argument("--train_path",default="8_kaggle房价预测\model\dataset\\1\\train.csv",help="定义k折交叉验证")
    parse.add_argument("--test_path",default="8_kaggle房价预测\model\dataset\\1\\test.csv",help="定义k折交叉验证")
    
    parse.add_argument("--hidden1",default=22,type=int,help="hidden1")
    parse.add_argument("--dropout1",default=0.2,type=float,help="dropout1")
    parse.add_argument("--hidden2",default=100,type=int,help="hidden2")
    parse.add_argument("--dropout2",default=0.2,type=float,help="dropout2")

    args=parse.parse_args()  #解析参数对象获取解析对象
    return args

'''数据集预处理'''
def data_pro(train_path,test_path):
    train_data=pd.read_csv(train_path)
    test_data=pd.read_csv(test_path)
    #将测试机和训练集合合并
    all_data=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))  #去除了价格
    '''处理数值型数据'''
    numeric_features=all_data.dtypes[all_data.dtypes!='object'].index  #数值类型的数据的属性名
    #no_numeric_features=all_data.dtypes[all_data.dtypes=='object'].index  #非数值类型
    all_data[numeric_features] = all_data[numeric_features].apply(        #将数值型数据归一化
            lambda x: (x - x.mean()) / (x.std()))
    #在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
    all_data[numeric_features] =all_data[numeric_features].fillna(0)
    '''处理离散型数据'''
    #“Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
    all_data=pd.get_dummies(all_data,dummy_na=True)
    '''划分测试机和训练集'''
    n_train=train_data.shape[0]
    train_f=torch.tensor(all_data[:n_train].values,dtype=torch.float32)  #训练数据
    test_f=torch.tensor(all_data[n_train:].values,dtype=torch.float32)   #测试数据
    train_label=torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)  #标签
    #返回数据迭代器和输入特征,训练集数据和标签
    return train_f.shape[1],train_f,train_label,test_f,test_data
    
'''形成数据迭代器'''
def load_array(batch_size,train_f,train_label):
    dataset=data.TensorDataset(train_f,train_label)
    train_iter=data.DataLoader(dataset,batch_size,shuffle=True)
    return train_iter

'''定义模型'''
#一个简单的线性模型作为Baseline
def model(in_features,hidden1,hidden2,dropout1,dropout2):
    net=torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features,hidden1),
        nn.ReLU(),
        nn.Linear(hidden1,1),
    )
    return net

'''模型评判'''
def log_rmse(net,train_f,train_label,loss):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1，将数据控制在1~正无穷
    clip_pred=torch.clamp(net(train_f),1,float("inf"))  #预测的数值
    rmse=torch.sqrt(loss(torch.log(clip_pred),torch.log(train_label)))  #均方误差
    return rmse.item()

'''自定义每个批次训练函数'''
def train(num_epochs,net,train_f,train_label,valid_f,valid_label,loss,optimizer,batch_size):
    #判断是不是pytorch得model，如果是，就打开训练模式，pytorch得训练模式默认开启梯度更新
    if isinstance(net,torch.nn.Module):
        net.train()
    #数据迭代器
    train_iter=load_array(batch_size,train_f,train_label)
    train_ls,valid_ls=[],[]
    #创建样本累加器【累加每批次的损失值、样本预测正确的个数、样本总数】
    for epoch in range(num_epochs):
        for x,y in train_iter:
            #前向传播获取预测结果
            y_hat=net(x)
            #计算损失
            l=loss(y_hat,y) 
            #判断是pytorch自带得方法还是我们手写得方法（根据不同得方法有不同得处理方式）
            if isinstance(optimizer,torch.optim.Optimizer):
                #梯度清零
                optimizer.zero_grad()
                #损失之求和，反向传播（pytorch自动进行了损失值计算）
                l.backward()
                #更新梯度
                optimizer.step()
        #计算训练相对误差，作为模型的评判标准
        train_ls.append(log_rmse(net,train_f,train_label,loss))  
        #计算测试的相对误差
        valid_ls.append(log_rmse(net,valid_f,valid_label,loss))
    
    return train_ls,valid_ls

'''k折交叉验证'''
#划分每一折的k折数据
def get_k_fold_data(k,i,train_f,train_label):
    assert k>1
    fold_size=train_f.shape[0]//k  #整除，每一折的数据大小
    x_train,y_train=None,None
    #划分训练集和测试集
    for j in range(k):
        '''获取索引'''
        indx=slice(j*fold_size,(j+1)*fold_size)  #100个数据，5折，每折20个数据
        x_part,y_part=train_f[indx,:],train_label[indx]
        #验证集
        if j==i:
            x_valid,y_valid=x_part,y_part
        #训练集
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            '''
            torch.cat([x_train,x_part],0)；0代表竖直拼接，1代表横向拼接
            '''
            x_train=torch.cat([x_train,x_part],0)  
            y_train=torch.cat([y_train,y_part],0)
        #返回测试集和训练集

    return x_train,y_train,x_valid,y_valid

'''k折验证求平均数值'''
def k_fold(k,num_epochs,net,train_f,train_label,loss,optimizer,batch_size):
    train_l_sum,valid_l_sum=0,0
    l_train,l_valid=[],[]
    for i in range(k):
        #获取数据集
        train_f,train_label,valid_f,valid_label=get_k_fold_data(k,i,train_f,train_label)
        #进行训练,获取每折的相对误差
        train_ls,valid_ls=train(num_epochs,net,train_f,train_label,valid_f,valid_label,loss,optimizer,batch_size)
        train_l_sum+=train_ls[-1]  #获取最后一次的相对误差
        valid_l_sum+=valid_ls[-1]
        l_train.extend(train_ls)
        l_valid.extend(valid_ls)
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
            f'验证log rmse{float(valid_ls[-1]):f}')
        if i==0:
            draw(num_epochs,train_ls,valid_ls)
    return train_l_sum/k,valid_l_sum/k

'''测试'''
def test(test_f,test_data):
    if isinstance(net,torch.nn.Module):
        net.eval()  #将模型设置为评估模式
    #预测数据
    preds=net(test_f).detach().numpy()  #将tensor转为numpy格式  
    test_data['SalePrice']=pd.Series(preds.reshape(1, -1)[0])
    submission=pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)  #横向合并
    submission.to_csv('8_kaggle房价预测\pred_result\submission.csv', index=False)

'''可视化'''
def draw(num_epochs,l_train,l_valid):
    fig,ax=plt.subplots()   #定义画布
    ax.grid(True)          #添加网格
    ax.set_xlabel("epoch")
    ax.set_ylabel("rmse")
    #ax.set_ylim(0,1)

    ax.plot(range(num_epochs),l_train,dashes=[6, 2],label="train")
    ax.plot(range(num_epochs),l_valid,dashes=[6, 2],label="valid")
    ax.legend()
    plt.show()

if __name__=="__main__":
    args=parse_args()
    #数据集预处理
    in_features,train_f,train_label,test_f,test_data=data_pro(args.train_path,args.test_path)
    print(in_features)
    #模型
    net=model(in_features,args.hidden1,args.hidden2,args.dropout1,args.dropout2)
    #相关组件
    optimizer=torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.w_decay)  #这里利用权重衰退，控制模型参数
    loss=torch.nn.MSELoss()
    #k折交叉验证
    train_l,valid_l=k_fold(args.k,args.num_epochs,net,train_f,train_label,loss,optimizer,args.batch_size)
    print(f'{args.k}-折验证: 平均训练log rmse: {float(train_l):f}, '
            f'平均验证log rmse: {float(valid_l):f}')
    #保存训练日志
    with open("8_kaggle房价预测\log\log.txt",mode="a",encoding="utf-8")as f:
        f.write(f'{args.k}-折验证: 平均训练log rmse: {float(train_l):f}, '
            f'平均验证log rmse: {float(valid_l):f},\n'
            f"参数:\nlr={args.lr},w_decay={args.w_decay},num_epochs={args.num_epochs},batch_size={args.batch_size},k={args.k}\
,hidden1={args.hidden1}\n")
    
    #模型测试
    #test(test_f,test_data)