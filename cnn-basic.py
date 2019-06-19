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
class ConvNet(torch.nn.Module):
    def __init__(self,num_classes):
        super(ConvNet, self).__init__()
        # 1*28*28 --> 8*28*28
        self.conv_1 = torch.nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1)
        # 8*28*28 --> 8*14*14
        self.pool_1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        # 8*14*14 --> 16*14*14
        self.conv_2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        # 16*14*14 --> 16*7*7
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #
        self.linear_1 = nn.Linear(16*7*7,num_classes)


    def forward(self, x):
        #第一层卷积
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)
        # 第二层卷积
        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)

        logits = self.linear_1(out.view(-1,16*7*7))
        probas = F.softmax(logits,dim=1)


        return logits,probas


model = ConvNet(num_classes=num_classes) #模型的初始化
model = model.cuda()  #放入GPU中
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#测试例子
def computer_accuracy(net,data_loader):
    net.eval()  #转换网络的模式
    correct_pred,num_example = 0,0
    with torch.no_grad():  #没有梯度
        for x,y in data_loader:
            x = x.cuda()
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
        x = x.cuda()
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

