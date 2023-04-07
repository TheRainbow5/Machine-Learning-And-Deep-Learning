from typing import Tuple
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils  import data 
import  argparse

import matplotlib.pyplot as plt


'''定义参数'''
def parse_args():
    parse=argparse.ArgumentParser(description="神经网络超参数")  #创建参数对象
    parse.add_argument("--lr",default=0.2,type=float,help="学习率")
    parse.add_argument("--num_epochs",default=20,type=int,help="训练次数")
    parse.add_argument("--batch_size",default=64,type=int)
    parse.add_argument("--resize",default=None,type=int,help="图片像素")
    
    parse.add_argument("--num_inputs",default=28*28,type=int,help="输入神经元个数")
    parse.add_argument("--hidden1",default=256,type=int,help="hidden1")
    parse.add_argument("--dropout1",default=0.5,type=float,help="dropout1")
    parse.add_argument("--hidden2",default=100,type=int,help="hidden2")
    parse.add_argument("--dropout2",default=0.2,type=float,help="dropout2")
    parse.add_argument("--num_outputs",default=10,type=int,help="输入神经元个数")

    args=parse.parse_args()  #解析参数对象获取解析对象
    return args

'''定义权重'''
def  init_w(m):
    if type(m)==torch.nn.Linear:
        torch.nn.init.normal_(m.weight,std=0.01)

'''下载数据集'''
def load_data(batch_size,resize):
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0到1之间
    trans =[transforms.ToTensor()]
    #修改图片大小
    if resize:
        trans.insert(0,transforms.Resize(resize)) 
    trans=transforms.Compose(trans)
    #下载训练数据
    mnist_train = torchvision.datasets.FashionMNIST(
        root="datasets",  #保存的目录
        train=True,       #下载的是训练数据集
        transform=trans,   #得到的是pytorch的tensor，而不是图片
        download=True)  #从网上下载
    #下载测试数据
    mnist_test = torchvision.datasets.FashionMNIST(
        root="datasets", train=False, transform=trans, download=True)
    #装载数据集
    data_loader_train=data.DataLoader(dataset=mnist_train,
                                                    batch_size=batch_size,
                                                    shuffle=True)   #数据是否打乱
    data_loader_test=data.DataLoader(dataset=mnist_test,
                                        batch_size=64,
                                        shuffle=True)
    #返回训练集和测试集
    return data_loader_train,data_loader_test

'''dropout_layer'''
def dropout_layer(x,dropout):
    assert 0<= dropout <=1   #True继续执行下面的程序，反之，异常结束程序
    #全部神经元丢弃
    if dropout==1:  
        return torch.zeros_like(x) 
    #所有神经元保留
    if dropout==0:
        return x
    '''
    torch.rand(x.shape):0~1之间的均匀分布
    torch.rand(x.shape)>dropout:返回一个True和False的数据
    (torch.rand(x.shape)>dropout).float():将True和False转为1和0
    '''
    mask=(torch.rand(x.shape)>dropout).float() 
    return mask*x/(1-dropout)  #返回dropout后的数据

class Net(torch.nn.Module):
    def __init__(self,inputs_shape,outputs_shape,hiddens1,hiddens2):
        super(Net,self).__init__()
        self.inputs_shape=inputs_shape
        self.lin1=torch.nn.Linear(inputs_shape,hiddens1)
        self.lin2=torch.nn.Linear(hiddens1,hiddens2)
        self.lin3=torch.nn.Linear(hiddens2,outputs_shape)
        self.relu=torch.nn.ReLU()
    #前向传播
    def forward(self,x,dropout1,dropout2,is_training):
        f=x.reshape((-1,self.inputs_shape))  #将数据展平
        h1=self.relu(self.lin1(f))  #hidden layer1
        #只有在训练模型时在是哟个dropout
        if is_training==True:
            h1=dropout_layer(h1,dropout1)
        h2=self.relu(self.lin2(h1))  #hidden layer2
        #只有在训练模型时在是哟个dropout
        if is_training==True:
            h2=dropout_layer(h2,dropout2)
        out=self.lin3(h2)
        return out

'''定义预测准确率函数'''
def acc(y_hat,y):
    '''
    :param y_hat: 接收二维张量，例如 torch.tensor([[1], [0]...])
    :param y: 接收二维张量，例如 torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]...]) 三分类问题
    :return:
    '''
    y_hat=y_hat.argmax(axis=1)  #获取概率最大得下标,就是判断得类别
    cmp=y_hat.type(y.dtype)==y  #数据类型是否相同
    return float(cmp.type(y.dtype).sum())
    
'''存储相关数据'''
class Accumulator():
    ''' 对评估的正确数量和总数进行累加 '''
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]

'''自定义每个批次训练函数'''
def train_epoch_cha3(net,data_loader_train,loss,optimizer,dropout1,dropout2):
    #判断是不是pytorch得model，如果是，就打开训练模式，pytorch得训练模式默认开启梯度更新
    if isinstance(net,torch.nn.Module):
        net.train()
    #创建样本累加器【累加每批次的损失值、样本预测正确的个数、样本总数】
    metric = Accumulator(3)  
    for x,y in data_loader_train:
        #前向传播获取预测结果
        y_hat=net.forward(x,dropout1,dropout2,True)
        #计算损失
        l=loss(y_hat,y) 
        #判断是pytorch自带得方法还是我们手写得方法（根据不同得方法有不同得处理方式）
        if isinstance(optimizer,torch.optim.Optimizer):
            #初始化梯度
            optimizer.zero_grad()
            #损失之求和，反向传播（pytorch自动进行了损失值计算）
            l.backward()
            #更新梯度
            optimizer.step()
            #累加个参数
            metric.add(
                float(l)*len(y),  #损失值总数
                acc(y_hat,y),     #计算预测正确得总数
                y.size().numel()  #样本总数
            )
    #返回平均损失值，预测正确得概率
    return metric[0]/metric[2],metric[1]/metric[2]

'''模型测试'''
def test_cha3(net,test_iter,dropout1,dropout2):
    if isinstance(net,torch.nn.Module):
        net.eval()  #将模型设置为评估模式
    metric=Accumulator(2)
    for x,y in test_iter:
        metric.add(
            acc(net.forward(x,dropout1,dropout2,False),y),  #计算准确个数
            y.numel()  #测试样本总数
        )
    #返回模型得准确率
    print(f"test_acc={metric[0]/metric[1]:.2f}%")
    return metric[0]/metric[1]
    
'''正式训练'''
def train_cha3(num_epochs,net,train_iter,test_iter,loss,optimizer,dropout1,dropout2):
    loss_list=[]
    train_acc=[]
    test_acc=[]
    for epoch in range(num_epochs):
        #计算训练数据的平均损失值和正确率
        train_metrics=train_epoch_cha3(net,train_iter,loss,optimizer,dropout1,dropout2)
        loss_list.append(train_metrics[0])  #保存loss
        train_acc.append(train_metrics[1])   #保存准确率
        #计算验证集的准确率
        test_metrics=test_cha3(net,test_iter,dropout1,dropout2)
        test_acc.append(test_metrics)
        print(f"epoch{epoch+1}:loss={train_metrics[0]},train_acc={train_metrics[1]*100:.2f}%,test_acc={test_metrics:.2f}%")
                
    return loss_list,train_acc,test_acc

'''可视化'''
def draw(num_epochs,loss_list,train_acc,test_acc):
    fig,ax=plt.subplots()   #定义画布
    ax.grid(True)          #添加网格
    ax.set_xlabel("epoch")
    ax.set_ylim(0,1)

    ax.plot(range(num_epochs),loss_list,label="loss")
    ax.plot(range(num_epochs),train_acc,dashes=[6, 2],label="train")
    ax.plot(range(num_epochs),test_acc,dashes=[6, 2],label="test")
    ax.legend()
    plt.show()

if __name__=="__main__":
    args=parse_args()
    #获取数据集
    train_iter,test_iter=load_data(batch_size=args.batch_size,resize=args.resize)
    #定义模型
    net=Net(args.num_inputs,args.num_outputs,args.hidden1,args.hidden2)
    #初始化w
    net.apply(init_w) 
    #计算损失值
    loss=torch.nn.CrossEntropyLoss()  #交叉熵，不要设置参数
    #计算梯度
    optimizer=torch.optim.SGD(net.parameters(),lr=args.lr)
    '''训练'''
    loss_list,train_acc,test_acc=train_cha3(args.num_epochs,net,train_iter,test_iter,loss,optimizer,args.dropout1,args.dropout2)
    #可视化
    draw(args.num_epochs,loss_list,train_acc,test_acc)




