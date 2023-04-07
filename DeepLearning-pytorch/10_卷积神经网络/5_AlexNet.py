import torch
import torchvision
from torchvision import transforms
from torch.utils  import data 
import  argparse

import matplotlib.pyplot as plt

'''定义参数'''
def parse_args():
    parse=argparse.ArgumentParser(description="神经网络超参数")  #创建参数对象
    parse.add_argument("--lr",default=0.2,type=float,help="学习率")
    parse.add_argument("--num_epochs",default=10,type=int,help="训练次数")
    parse.add_argument("--batch_size",default=64,type=int)
    parse.add_argument("--resize",default=224,type=int,help="图片像素")
    args=parse.parse_args()  #解析参数对象获取解析对象
    return args

'''下载数据集'''
def load_data(batch_size,resize=None):
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
    len(mnist_train),len(mnist_test)
    #装载数据
    data_loader_train=data.DataLoader(dataset=mnist_train,
                                                batch_size=batch_size,
                                                shuffle=True)   #数据是否打乱
    data_loader_test=data.DataLoader(dataset=mnist_test,
                                    batch_size=batch_size,
                                    shuffle=True)
    return data_loader_train,data_loader_test
'''模型定义'''
def AlexNet():
    net=torch.nn.Sequential(
        #cnn1
        torch.nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3,stride=2),
        #cnn2
        torch.nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3,stride=2), 
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        torch.nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3,stride=2),
        torch.nn.Flatten(),
        #这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        torch.nn.Linear(6400,4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096,4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096,10)
    )
    return net

'''定义预测准确率函数'''
def acc(y_hat,y):
    '''
    :param y_hat: 接收二维张量，例如 torch.tensor([[1], [0]...])
    :param y: 接收二维张量，例如 torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]...]) 三分类问题
    :return:
    '''
    y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y  #数据类型是否相同
    return float(cmp.type(y.dtype).sum())
    
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
def train_epoch(net,train_iter,loss,optimizer,device):
    #判断是不是pytorch得model，如果是，就打开训练模式，pytorch得训练模式默认开启梯度更新
    if isinstance(net,torch.nn.Module):
        net.train()
    #创建样本累加器【累加每批次的损失值、样本预测正确的个数、样本总数】
    metric = Accumulator(3)  
    for x,y in train_iter:
        x=x.to(device)                            #<----------------------GPU
        y=y.to(device)
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
            #累加个参数
            metric.add(
                float(l)*len(y),  #损失值总数
                acc(y_hat,y),     #计算预测正确得总数
                y.size().numel()  #样本总数
            )
    #返回平均损失值，预测正确得概率
    return metric[0]/metric[2],metric[1]/metric[2]

'''模型测试'''
def test_epoch(net,test_iter,device):
    if isinstance(net,torch.nn.Module):
        net.eval()  #将模型设置为评估模式
    metric=Accumulator(2)
    for x,y in test_iter:
        x=x.to(device)                            #<----------------------GPU
        y=y.to(device)
        metric.add(
            acc(net.forward(x),y),  #计算准确个数
            y.numel()  #测试样本总数
        )
    return metric[0]/metric[1]

'''正式训练'''
def train_LeNet(num_epochs,trian_iter,test_iter,lr):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    net=AlexNet()
    net=net.to(device)  #将网络放在gpu或cpu运行<--------------
    loss_list=[]
    train_acc=[]
    test_acc=[]
    #初始化权重
    def init_weight(m):
        if type(m)==torch.nn.Linear or type(m)==torch.nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight)
    net.apply(init_weight)
    #损失函数
    loss=torch.nn.CrossEntropyLoss()
    #优化器
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    #训练
    for epoch in range(num_epochs):
        #返回平均损失值和正确率
        train_metrics=train_epoch(net,trian_iter,loss,optimizer,device)  #<-----训练
        loss_list.append(train_metrics[0])  #保存loss
        train_acc.append(train_metrics[1])   #保存准确率
        #测试集
        test_metric=test_epoch(net,test_iter,device)     #<-------------测试
        test_acc.append(test_metric)
        print(f"epoch{epoch+1}:loss={train_metrics[0]},train_acc={train_metrics[1]*100:.2f}%,test_acc={test_metric*100:.2f}%")
    
    return loss_list,train_acc,test_acc

'''可视化'''
def draw(num_epochs,loss_list,train_acc,test_acc):
    fig,ax=plt.subplots()   #定义画布
    ax.grid(True)          #添加网格
    ax.set_xlabel("epoch")
    #ax.set_ylim(0,1)

    ax.plot(range(num_epochs),loss_list,label="loss")
    ax.plot(range(num_epochs),train_acc,dashes=[6, 2],label="train")
    ax.plot(range(num_epochs),test_acc,dashes=[6, 2],label="test")
    ax.legend()
    plt.show()

if __name__=="__main__":
    args=parse_args()
    #数据集
    train_iter,test_iter=load_data(batch_size=args.batch_size,resize=args.resize)
    #训练
    loss_list,train_acc,test_acc=train_LeNet(args.num_epochs,train_iter,test_iter,args.lr)
    #可视化
    draw(args.num_epochs,loss_list,train_acc,test_acc)