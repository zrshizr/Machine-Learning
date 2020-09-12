import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision
import torchvision.transforms as transforms
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#参数 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100#很好理解吧
leaarning_rate = 0.001
#data
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

#数据加载
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#创建网络部分
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# model = NeuralNet(input_size, hidden_size, num_classes)    
model = NeuralNet(input_size, hidden_size, num_classes)

#选择Loss_function
criterion = nn.CrossEntropyLoss()
#优化器
optimizer = torch.optim.Adam(model.parameters(),lr=leaarning_rate)

#开始训练模型
for i ,(images,lable) in enumerate(train_loader):
    images = images.reshape(-1, 28 * 28)
    #前向传播
    outputs = model(images)
    loss = criterion(outputs, lable)
    #反向
    optimizer.zero_grad() #意思是把梯度置零，也就是把loss关于weight的导数变成0.
    loss.backward()
    optimizer.step()#计算模型中所有张量的梯度后,调用optimizer.step()会使优化器迭代它应该更新的所有参数(张量),并使用它们内部存储的grad来更新它们的值.

    # if (i + 1) % 100 == 0:
    #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
    #             .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    
