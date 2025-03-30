import numpy as np
import torch


def get_batch_loss(model, loss_func, xb, yb, optimizer=None, verbose=False):
    '''
    Apply loss function to a batch of inputs. If no optimizer
    is provided, skip the back prop step.
    '''
    if verbose:
        print('loss batch ****')
        print("xb shape:", xb.shape)
        print("yb shape:", yb.shape)
        print("yb shape:", yb.squeeze(1).shape)
        # print("yb",yb)

    # get the batch output from the model given your input batch
    # ** This is the model's prediction for the y labels! **
    xb_out = model(xb.float())

    if verbose:
        print("model out pre loss", xb_out.shape)
        # print('xb_out', xb_out)
        print("xb_out:", xb_out.shape)
        print("yb:", yb.shape)
        print("yb.long:", yb.long().shape)

    loss = loss_func(xb_out, yb.squeeze(1))

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), len(xb)


def train_step(model, train_dl, loss_func, device, opt):
    '''
    Execute 1 set of batched training within an epoch
    '''
    # Set model to Training mode
    model.train()
    tl = []  # train losses
    ns = []  # batch sizes, n

    # loop through train DataLoader
    for xb, yb in train_dl:
        # put on GPU
        xb, yb = xb.to(device), yb.to(device)

        # provide opt so backprop happens
        loss, batch_size = get_batch_loss(model, loss_func, xb, yb, optimizer=opt)

        # collect train loss and batch sizes
        tl.append(loss)
        ns.append(batch_size)

    # average the losses over all batches
    train_loss = np.sum(np.multiply(tl, ns)) / np.sum(ns)

    return train_loss


def val_step(model, val_dl, loss_func, device):
    '''
    Execute 1 set of batched validation within an epoch
    '''
    # Set model to Evaluation mode
    model.eval()
    with torch.no_grad():
        vl = []  # val losses
        ns = []  # batch sizes, n

        # loop through validation DataLoader
        for xb, yb in val_dl:
            # put on GPU
            xb, yb = xb.to(device), yb.to(device)

            # Do NOT provide opt here, so backprop does not happen
            loss, batch_size = get_batch_loss(model, loss_func, xb, yb)

            # collect val loss and batch sizes
            vl.append(loss)
            ns.append(batch_size)

    # average the losses over all batches
    val_loss = np.sum(np.multiply(vl, ns)) / np.sum(ns)

    return val_loss


def fit(epochs, model, loss_func, opt, train_dl, val_dl, device, patience=1000):
    '''
    Fit the model params to the training data, eval on unseen data.
    Loop for a number of epochs and keep train of train and val losses
    along the way
    '''
    # keep track of losses
    train_losses = []
    val_losses = []

    # loop through epochs
    for epoch in range(epochs):
        # take a training step
        train_loss = train_step(model, train_dl, loss_func, device, opt)
        train_losses.append(train_loss)

        # take a validation step
        val_loss = val_step(model, val_dl, loss_func, device)
        val_losses.append(val_loss)

        print(f"E{epoch} | train loss: {train_loss:.3f} | val loss: {val_loss:.3f}")

    return train_losses, val_losses


def run_model(train_dl, val_dl, model, device,
              lr=0.001, epochs=15):
    '''
    Given train and val DataLoaders and a NN model, fit the mode to the training
    data.
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_func = torch.nn.CrossEntropyLoss()

    # run the training loop
    train_losses, val_losses = fit(
        epochs,
        model,
        loss_func,
        optimizer,
        train_dl,
        val_dl,
        device)

    return train_losses, val_losses
