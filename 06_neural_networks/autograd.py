import torch, torchvision
import numpy as np

# torch.autograd is PyTorch's automatic differentiation engine that powers neural network training

# load pretrained resnet18 model from torchvision
model = torchvision.models.resnet18(pretrained=True)
# create a random data tensor to represent a single image
# with 3 channels, height and width of 64, and its labels initialized to random values
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# make the forward pass
prediction = model(data)
# calculate error, backpropogate error through network
loss = (prediction - labels).sum()
loss.backward()

# load optimizer:
# stochastic gradient descent, learning rate 0.01, momentum 0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

## Differentiation in Autogard
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)

# create another tensor q from a and b
Q = 3 * a ** 3 - b ** 2

# assume a and b to be parameters of a NN, Q its error
# want the gradients w.r.t parameters
external_grad = torch.tensor([1.0, 1.0])
Q.backward(gradient=external_grad)
print(9 * a ** 2 == a.grad)
print(-2 * b == b.grad)
