
import torch


# sigmoid activation function using pytorch
def sigmoid_activation(z):
    return 1 / (1 + torch.exp(-z))

# function to calculate the derivative of activation
def sigmoid_delta(x):
  return x * (1 - x)


n_input, n_hidden, n_output = 5,3,1

x = torch.randn((1, n_input))
y = torch.rand((1, n_output))

# hidden and output layer weights
w1 = torch.randn(n_input,n_hidden)
w2 = torch.randn(n_hidden, n_output)

# Initialize tensor variables for bias terms
b1 = torch.randn((1,n_hidden))  # Bias for hidden layer
b2 = torch.randn((1,n_output))  # bias for output layer

print("Old bias terms")
print(b1)
print(b2)

# activation of hidden layer
z1 = torch.mm(x, w1) + b1
a1 = sigmoid_activation(z1)

# activation (output) of final layer
z2 = torch.mm(a1, w2) + b2
output = sigmoid_activation(z2)

loss = y - output


# compute derivative of error terms
delta_output = sigmoid_delta(output)
delta_hidden = sigmoid_delta(a1)

# backpass the changes to previous layers
d_outp = loss * delta_output
loss_h = torch.mm(d_outp, w2.t())
d_hidn = loss_h * delta_hidden

learning_rate = 0.3

w2 += torch.mm(a1.t(), d_outp) * learning_rate
w1 += torch.mm(x.t(), d_hidn) * learning_rate

b2 += d_outp.sum() * learning_rate
b1 += d_hidn.sum() * learning_rate

print("new bias terms")
print(b1)
print(b2)





