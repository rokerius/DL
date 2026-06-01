[Источник 1](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) [Источник 2](https://huggingface.co/blog/dvgodoy/beginner-pytorch-tutorial)
![[Pasted image 20260601140947.png|531]]

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
# numpy.ndarray -> torch.Tensor
```

#### Creating Parameters
```python
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
# при переносе на другой device градиенты теряются
# поэтому правильнее будет так:
a = torch.randn(1, dtype=torch.float).to(device)
b = torch.randn(1, dtype=torch.float).to(device)
a.requires_grad_()
b.requires_grad_()

# но еще лучше так:
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
```
*In PyTorch, every method that ends with an '_' makes changes in-place*

#### Autograd
```python
for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()     
    
    # We just tell PyTorch to work its way BACKWARDS from the specified loss!
    loss.backward()
    
    
    # AttributeError: 'NoneType' object has no attribute 'zero_'
    # a = a - lr * a.grad
    # b = b - lr * b.grad (при -= тоже ошибка)
    
    # We need to use NO_GRAD to keep the update out of the gradient computation
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    a.grad.zero_()
    b.grad.zero_()
```

#### Dynamic Computation Graph
[Подробнее лучше почитать тут]([obsidian://open?vault=Study&file=DL%2FDL%20Intro%2FNeural%20Networks](obsidian://open?vault=Study&file=Convex%20Optimization%2F%D0%90%D0%B2%D1%82%D0%BE%D0%BC%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5%20%D0%B4%D0%B8%D1%84%D1%84%D0%B5%D1%80%D0%B5%D0%BD%D1%86%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5.pdf))
```python
yhat = a + b * x_train_tensor
error = y_train_tensor - yhat
loss = (error ** 2).mean()
```

![[Pasted image 20260601144137.png|486]]
#### Model
```python
class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
    def forward(self, x):
        return self.a + self.b * x
```

```python
model = ManualLinearRegression().to(device)
lr = 1e-1
n_epochs = 1000
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train()  # не тренит, а переводит в "режим"
    yhat = model(x_train_tensor)
    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

```

#### Nested Models
```python
class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
                
    def forward(self, x):
        return self.linear(x)
```

#### Sequential Models
```python
model = nn.Sequential(nn.Linear(1, 1)).to(device)
```

#### Training step
```python
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

lr = 1e-1
n_epochs = 1000
model = nn.Sequential(nn.Linear(1, 1)).to(device)
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

train_step = make_train_step(model, loss_fn, optimizer)
losses = []

for epoch in range(n_epochs):
    loss = train_step(x_train_tensor, y_train_tensor)
    losses.append(loss)
```
Это больше косметика уже

#### Dataset
