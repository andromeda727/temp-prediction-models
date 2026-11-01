import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np

def train(epochs:int, model:nn.Module, xdata:torch.Tensor, ydata:torch.Tensor, scaler:StandardScaler, loss_func:int = 1, opt_method:int = 1, lr:float = 0.01, diagnostics: bool = True) -> tuple[list, list]:
    '''Train the given model using the given data
    Params:
        epochs: # of epochs to train for
        model: model to train
        xdata: training data to use for prediction
        ydata: true training values to predict against
        scaler: scaler used for normalization of target values, if any
        loss_func: integer taking values 1 or 2 to choose loss function -- 1=MSE, 2=Huber
        opt_method: integer taking values 1 or 2 to choose optimization method -- 1=SGD, 2=Adam
        lr: learning rate
    Returns:
        Two lists of losses -- one loss value for every 20th epoch. First list is normalized MSE, second list is raw MSE (MW^2)'''

    #Construct the loss function
    match loss_func:
        case 1: criterion = torch.nn.MSELoss()
        case 2: criterion = torch.nn.HuberLoss()
        case _: raise ValueError("Invalid loss function passed, please pass 1 or 2")

    
    # Construct the optimizer (Stochastic Gradient Descent in this case)
    match opt_method:
        case 1: optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        case 2: optimizer = torch.optim.Adam(model.parameters(), lr = lr) # lr is learning rate
        case _: raise ValueError("Invalid optimization method passed, please pass 1 or 2")

    losses_curve_norm = []
    losses_curve_raw = []

    # Gradient Descent
    loss = None
    for epoch in range(epochs):
        # Forward pass: Compute predicted y by passing x to the model
        Y_pred = model(xdata)
        # print(Y_pred.shape)
        # Compute and print loss
        if scaler is not None and scaler.scale_ is not None and scaler.mean_ is not None:
            Y_pred_raw = Y_pred * scaler.scale_[0] + scaler.mean_[0]
            Y_actual_raw = ydata * scaler.scale_[0]+ scaler.mean_[0]
            losses_curve_raw.append(criterion(torch.tensor(Y_pred_raw, dtype=torch.float32), torch.tensor(Y_actual_raw, dtype=torch.float32)))
        loss = criterion(Y_pred, ydata)
        losses_curve_norm.append(loss)
        if (diagnostics and epoch % 20 == 0):
            print('epoch: ', epoch,' loss: ', loss.item())
        

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()
    losses_curve_norm.append(loss)
    return losses_curve_norm, losses_curve_raw