import torchvision
import torch
from torchvision import transforms
import torch.optim as optim
from  torch.utils  import data 
import  argparse

def parse_args():
    parse=argparse.ArgumentParser(description="神经网络超参数")  #创建参数对象
    parse.add_argument("--lr",default=0.2,type=float,help="学习率")
    parse.add_argument("--num_epochs",default=30,type=int,help="训练次数")
    parse.add_argument("--num_inputs",default=28*28,type=int,help="输入神经元个数")
    parse.add_argument("--num_outputs",default=10,type=int,help="输入神经元个数")
    #对比三种激活函数（torch.nn.Relu、torch.nn.Tanh、torch.nn.Sigmoid）
    parse.add_argument("--activation_fuction",default=torch.nn.ReLU(),help="激活函数")
    args=parse.parse_args()  #解析参数对象获取解析对象
    return args

'''数据集下载及处理'''
def load_data():
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0到1之间
    trans = transforms.ToTensor()
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
                                                    batch_size=64,
                                                    shuffle=True)   #数据是否打乱
    data_loader_test=data.DataLoader(dataset=mnist_test,
                                        batch_size=64,
                                        shuffle=True)
    #返回训练集和测试集
    return data_loader_train,data_loader_test

'''定义权重'''
def  init_w(m):
    if type(m)==torch.nn.Linear:
        torch.nn.init.normal_(m.weight,std=0.01)

'''定义模型'''
#神经网络
def model(num_inputs,num_outputs,activation_fuction):
    net=torch.nn.Sequential(
        torch.nn.Flatten(),   #将数据展平s
        torch.nn.Linear(num_inputs,20),  #隐藏层
        activation_fuction,
        torch.nn.Linear(20,num_outputs)  #输出层
    )
    return net

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
    
'''存储相关是数据'''
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
def train_epoch_cha3(net,data_loader_train,loss,optimizer):
    #判断是不是pytorch得model，如果是，就打开训练模式，pytorch得训练模式默认开启梯度更新
    if isinstance(net,torch.nn.Module):
        net.train()
    #创建样本累加器【累加每批次的损失值、样本预测正确的个数、样本总数】
    metric = Accumulator(3)  
    for x,y in data_loader_train:
        #前向传播获取预测结果
        y_hat=net(x)
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

'''正式训练'''
def train_cha3(num_epochs,net,data_loader_train,loss,optimizer):
    for epoch in range(num_epochs):
        #返回平均损失值和正确率
        train_metrics=train_epoch_cha3(net,data_loader_train,loss,optimizer)
        print(f"epoch{epoch+1}:loss={train_metrics[0]},acc={train_metrics[1]*100:.2f}%")

'''测试模型'''
def test_cha3(net,test_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()  #将模型设置为评估模式
    metric=Accumulator(2)
    for x,y in test_iter:
        metric.add(
            acc(net(x),y),  #计算准确个数
            y.numel()  #测试样本总数
        )
    #返回模型得准确率
    print(f"test_acc={metric[0]/metric[1]:.2f}%")
    return metric[0]/metric[1]

if __name__=="__main__":
    args=parse_args()
    net=model(args.num_inputs,args.num_outputs,args.activation_fuction)
    #初始化w
    net.apply(init_w) 
    #计算损失值
    loss=torch.nn.CrossEntropyLoss()  #交叉熵，不要设置参数
    #计算梯度
    optimizer=torch.optim.SGD(net.parameters(),lr=args.lr)
    #获取数据集
    train_iter,test_iter=load_data()
    '''训练'''
    train_cha3(args.num_epochs,net,train_iter,loss,optimizer)
    '''测试'''
    test_cha3(net,test_iter)

