
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#设置超参数

input_size  = 1
output_size = 1
num_epoches = 50
learning_rate = 0.001

# 数据
x_train  = np.array([[1.1],[3.2],[3.1],[2.8],[4.1],[5.9],[3.0]],dtype=np.float32)
y_train  = np.array([[1.11],[3.0],[2.9],[2.6],[3.8],[6.0],[3.1]],dtype=np.float32)
# plt.plot(x_train,y_train,'ro')
# plt.show()

model = nn.Linear(input_size,output_size)
credits = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr =learning_rate)
#训练
for epoche in range(5):
    for i in range(num_epoches):
        #转为torch的格式
        input = torch.from_numpy(x_train)
        lable = torch.from_numpy(y_train)
        #前向传播
        outputs = model(input)
        loss = credits(outputs,lable)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(i+1, num_epoches, loss.item()))
    #画图
predicated = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicated, label='Fitted line')
plt.legend()
plt.show()




