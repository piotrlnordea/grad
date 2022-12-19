import numpy as np
import pandas as pd
import torch

import torch.optim as optim

import torch.nn as nn
from torch.autograd import Variable

#
# set seed for repeatability
#
np.random.seed(68)  # bureau 68
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)

#
#  input to the neural network
#
t_c = [1.0]
t_u = [1.0]
t_c = np.array(t_c)
t_u = np.array(t_u)


def numpy_to_variable(r):
    return Variable(torch.from_numpy(r).float(), requires_grad=True).to(device)


#
#  define the netowrk configuration
#
model_nn = nn.Sequential(
    nn.Linear(1, 2),  # input layer
    nn.ReLU(),
    nn.Linear(2, 2),  # hidden layer
    nn.ReLU(),
    nn.Linear(2, 1)  # output layer
).to(device)

#
#  Do 2 epochs to check updated weights and biases
#
n_epochs = 223
learning_rate = 1e-2
optL = optim.SGD(model_nn.parameters(), lr=learning_rate)

loss_fn = torch.nn.MSELoss(size_average=False, reduce=False, reduction='sum')

t1_u = numpy_to_variable(t_u[:])
t_ucol = torch.tensor(t1_u, device=device, requires_grad=True)
t1_c = numpy_to_variable(t_c[:])
t_ccol = torch.tensor(t1_c, device=device, requires_grad=True)

for epochs in range(n_epochs):

    t_u1 = torch.cat([t_ucol.view((-1, 1))], dim=1)
    t_c1 = torch.cat([t_ccol.view((-1, 1))], dim=1)

    #
    #  Extract the random weights and biases from the model
    #

    wt_L0 = model_nn[0].weight.detach().to('cpu').numpy()
    bias_L0 = model_nn[0].bias.detach().to('cpu').numpy()

    wt_L2 = model_nn[2].weight.detach().to('cpu').numpy()
    bias_L2 = model_nn[2].bias.detach().to('cpu').numpy()

    wt_L4 = model_nn[4].weight.detach().to('cpu').numpy()
    bias_L4 = model_nn[4].bias.detach().to('cpu').numpy()

    print(f"epoch:{epochs}, weight_input_layer: {wt_L0}")
    print(f"epoch:{epochs}, bias_input_layer: {bias_L0}")
    print(" ")

    print(f"epoch:{epochs}, weight_hidden_layer: {wt_L2}")
    print(f"epoch:{epochs}, bias_hidden_layer: {bias_L2}")
    print(" ")

    print(f"epoch:{epochs}, weight_output_layer: {wt_L4}")
    print(f"epoch:{epochs}, bias_output_layer: {bias_L4}")
    print(" ")
    #
    #  print output from forward feed
    #

    t_output = model_nn(t_u1)
    print("output:", t_output)

    loss_train = loss_fn(t_output, t_c1)
    print("loss:", loss_train)
    print(" ")
    #
    #  zero out gradients and compute gradients of loss with respect to
    #  weights and biases.
    #
    optL.zero_grad()
    loss_train.backward(retain_graph=True)

    gradw_L0 = model_nn[0].weight.grad.detach().to('cpu').numpy()
    gradb_L0 = model_nn[0].bias.grad.detach().to('cpu').numpy()

    gradw_L2 = model_nn[2].weight.grad.detach().to('cpu').numpy()
    gradb_L2 = model_nn[2].bias.grad.detach().to('cpu').numpy()

    gradw_L4 = model_nn[4].weight.grad.detach().to('cpu').numpy()
    gradb_L4 = model_nn[4].bias.grad.detach().to('cpu').numpy()

    print(f"epoch:{epochs}, grad_w_input_layer: {gradw_L0}")
    print(f"epoch:{epochs}, grad_b_input_layer: {gradb_L0}")
    print(" ")

    print(f"epoch:{epochs}, grad_w_hidden_layer: {gradw_L2}")
    print(f"epoch:{epochs}, grad_b_hidden_layer: {gradb_L2}")
    print(" ")

    print(f"epoch:{epochs}, gradw_output_layer: {gradw_L4}")
    print(f"epoch:{epochs}, gradb_output_layer: {gradb_L4}")
    print(" ")

    curr_lr = optL.param_groups[0]["lr"]
    print("learning rate:", curr_lr)

    optL.step()

    print("###")