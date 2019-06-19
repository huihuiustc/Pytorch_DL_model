import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
#参数配置
learning_rate = 0.1
batch_size = 256
epochs = 10
acc = 0.5
num_features = 28*28
num_hidden_1 = 256
num_hidden_2 = 128
num_classes = 10
#训练集和测试集的读取
train_data = datasets.MNIST(root='data',train=True,transform=transforms.ToTensor(),download=False)
test_data = datasets.MNIST(root='data',train=False,transform=transforms.ToTensor(),download=False)
#训练集和测试集的加载
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)
#训练集的尺寸预览
for image,labels in train_loader:
    print('图片的尺寸',image.shape)  #torch.Size([1, 1, 28, 28])
    print('标签的尺寸',labels.shape) #torch.Size([1])
    break
#多层感知机model的搭建
class MultilayerPerceptron(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super(MultilayerPerceptron, self).__init__()
        #第一层
        self.linear_1 = nn.Linear(num_features,num_hidden_1)
        self.linear_1.weight.detach().normal_(0.0,0.1)# initialization
        self.linear_1.bias.detach().zero_()  # initialization
        #第二层
        self.Linear_2 = nn.Linear(num_hidden_1,num_hidden_2)
        self.Linear_2.weight.detach().normal_(0.0, 0.1)
        self.Linear_2.bias.detach().zero_()
        #第三层 输出层
        self.Linear_out = nn.Linear(num_hidden_2,num_classes)
        self.Linear_out.weight.detach().normal_(0.0, 0.1)
        self.Linear_out.bias.detach().zero_()
    def forward(self, x):
        out = self.linear_1(x)
        out = F.relu(out)
        out = self.Linear_2(out)
        out = F.relu(out)
        out = self.Linear_out(out)
        probas = F.log_softmax(out,dim=1)
        return out,probas
model = MultilayerPerceptron(num_features=num_features,num_classes=num_classes) #模型的初始化
model = model.cuda()  #放入GPU中
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#测试例子
def computer_accuracy(net,data_loader):
    net.eval()  #转换网络的模式
    correct_pred,num_example = 0,0
    with torch.no_grad():  #没有梯度
        for x,y in data_loader:
            x = x.view(-1,28*28).cuda()
            y = y.cuda()
            out,probas = net(x)
            _,predicted_labels = torch.max(probas,1)
            num_example += y.size(0)
            correct_pred += (predicted_labels == y).sum()
        return correct_pred.float()/num_example

start_time = time.time()
for epoch in range(epochs):
    model.train()
    for batch_idx,(x,y) in enumerate(train_loader):
        x = x.view(-1,28*28).cuda()
        y = y.cuda()
        #前向传播
        out, probas = model(x)
        loss = F.cross_entropy(out,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | loss: %.4f'
                  % (epoch + 1, epochs, batch_idx,
                     len(train_loader), loss))
    with torch.set_grad_enabled(False):
        test_loss = computer_accuracy(model, test_loader)
        print('Epoch: %03d/%03d 准确度: %.2f%%' % (
            epoch + 1, epochs,test_loss))
        #保存模型
        if test_loss.item() > acc:
            model_name = 'model_'+str(test_loss.item())+'.pth'
            torch.save(model.state_dict(),model_name)
            acc = test_loss.item()

    print('时间: %.2f min' % ((time.time() - start_time) / 60))
print('总共时间: %.2f min' % ((time.time() - start_time) / 60))

