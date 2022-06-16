import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
# to scale the features
from sklearn.preprocessing import StandardScaler
# to split the data into training and testing data
from sklearn.model_selection import train_test_split

# 0. Prepare data

# binary classification data on which we can predict breast cancer
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

n_samples,n_features = X.shape

# data splitting
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 1234)

# scale features to have 0 mean and unit variance(standard deviation = 1)
sc  =StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# convert numpy arrays to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# since y is a just one row vector, we convert into a column vector 
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1. Model
# f = wx+b , sigmoid at the end for logistic regression

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)

# 2. Loss and Optimizer
# we are using binary cross entropy loss and SGD optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# 3. Training loop
epochs = 100

for epoch in range(epochs):
    # forward pass and loss calculation
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    # backward_pass
    loss.backward()
    
    # update weights
    optimizer.step()
    
    # empty gradients to prevent weight accumulation
    optimizer.zero_grad()
    
    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        
# evaluate the model using test data
# evaluation doesn't need a computational graph and doesn't have the requirement to compute gradients. 
# So, we use torch.no_grad()
with torch.no_grad():
    y_pred = model(X_test)
    # if sigmoid outpyt -> y_pred > 0.5 then class label is 1 else 0
    y_pred_classes = y_pred.round()
    # function eq() is used to check if elements in two tensors are equal or not 
    # sum() function is used to add all elements of a tensor
    acc = y_pred_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
