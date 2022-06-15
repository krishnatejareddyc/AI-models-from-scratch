import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#prepare data
x,y = datasets.make_regression(n_samples=100,n_features=1,noise =20,random_state=1)

#convert data in numpy format to tensor fomat
X = torch.from_numpy(x.astype(np.float32))
Y = torch.from_numpy(y.astype(np.float32))


#reshape tensor to have one column 
Y = Y.view(Y.shape[0],1)

n_samples,n_features = X.shape

#model
input_size =n_features
output_size = 1
model = nn.Linear(input_size,output_size)

#loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

#training loop
epochs =  100
for epoch in range(epochs):
    #forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred,Y)
    
    #backward pass
    loss.backward()
    
    #update weights
    optimizer.step()
    
    # empty gradients to prevent gradient accumulation before next iteration
    optimizer.zero_grad()
    
    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}' )
    

# detach() used to prevent calculating gradients.
# calculate predictions on the computed final model
predicted = model(X).detach().numpy()

# plot
plt.plot(x,y,'ro')
plt.plot(x, predicted,'b')
plt.show()
