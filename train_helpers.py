from typing import Dict

import numpy as np
import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection


def compute_metrics(y_pred, y_true, y_probs, num_classes) -> Dict[str, float]:
    metrics = MetricCollection(
        {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
            "f1": F1Score(task="multiclass",num_classes=num_classes, average="macro"),
            "precision": Precision(task="multiclass",num_classes=num_classes, average="macro"),
            "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
        }
    )
    metrics.update(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32))
    metrics_computed = metrics.compute()
    metrics_computed = {k: v.item() for k, v in metrics_computed.items()}
    return metrics_computed

def train_batch(model, loss_func, x, y, optimizer=None):
    '''
    Apply loss function to a batch of inputs. If no optimizer
    is provided, skip the back prop step.
    '''
    # get the batch output from the model given your input batch
    # ** This is the model's prediction for the y labels! **
    logits = model(x.float())

    loss = loss_func(logits, y.squeeze(1))

    # calculate the predicted y labels
    predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), len(x), predictions, probabilities


def train_step(model, train_dl, loss_func, device, optimizer):
    '''
    Execute 1 set of batched training within an epoch.
    '''
    model.train()
    losses = []  # train losses
    batch_sizes = []  # batch sizes, n

    for x_batch, y_batch in train_dl:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        loss, batch_size, _, _ = train_batch(model, loss_func, x_batch, y_batch, optimizer=optimizer)

        losses.append(loss)
        batch_sizes.append(batch_size)

    # average the losses over all batches
    train_loss = np.sum(np.multiply(losses, batch_sizes)) / np.sum(batch_sizes)

    return train_loss


def eval_step(model, val_dl, loss_func, device):
    '''
    Execute 1 set of batched evaluation.
    '''
    model.eval()
    with torch.no_grad():
        losses = []  # val losses
        batch_sizes = []  # batch sizes, n
        preds = []
        probs = []
        true_labels = []

        for x_batch, y_batch in val_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            loss, batch_size, batch_preds, batch_probs = train_batch(model, loss_func, x_batch, y_batch, optimizer=None)

            losses.append(loss)
            batch_sizes.append(batch_size)
            true_labels.append(y_batch.squeeze(1).detach().cpu().numpy())
            preds.append(batch_preds)
            probs.append(batch_probs)

    # average the losses over all batches
    val_loss = np.sum(np.multiply(losses, batch_sizes)) / np.sum(batch_sizes)

    return val_loss, np.concatenate(preds), np.concatenate(probs), np.concatenate(true_labels)


def fit(epochs, model, loss_func, optimizer, train_dl, val_dl, num_classes, device, patience=1000):
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
        train_loss = train_step(model, train_dl, loss_func, device, optimizer)
        train_losses.append(train_loss)

        _, train_summary, _, _, _ = get_eval_summary(model=model, val_dl=train_dl,
                                                    loss_func=loss_func,
                                                    num_classes=num_classes,
                                                    device=device)
        print(f"E{epoch} | Train loss: {train_loss:.3f} | metrics: {train_summary}")

        # take a validation step
        val_loss, summary_message, _, _, _ = get_eval_summary(model=model, val_dl=val_dl, loss_func=loss_func, num_classes=num_classes, device=device)
        print(f"E{epoch} | Val: {summary_message}")

        val_losses.append(val_loss)

    return train_losses, val_losses


def get_eval_summary(model, val_dl, loss_func, num_classes, device):
    val_loss, epoch_preds, epoch_probs, epoch_labels = eval_step(model, val_dl,
                                                                 loss_func, device)
    # calculate the accuracy and classification metrics from the predictions and probs
    metrics = compute_metrics(y_pred=epoch_preds, y_true=epoch_labels,
                              y_probs=epoch_probs, num_classes=num_classes)
    summary_message = f"loss: {val_loss:.3f} | acc. {metrics['accuracy']:.3f} | f1: {metrics['f1']:.3f}"
    return val_loss, summary_message, epoch_preds, epoch_probs, epoch_labels


def run_model(train_dl, val_dl, test_dl, model, num_classes, device, lr=0.001, epochs=30):
    '''
    Given train and val DataLoaders and a NN model, fit the mode to the training
    data.
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # run the training loop
    train_losses, val_losses = fit(
        epochs=epochs,
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_dl=train_dl,
        val_dl=val_dl,
        num_classes=num_classes,
        device=device)

    # run the evaluation on the test set
    _, _, test_preds, test_probs, test_labels = get_eval_summary(model=model, val_dl=test_dl, loss_func=loss_func, num_classes=num_classes, device=device)

    return train_losses, val_losses, test_preds, test_probs, test_labels
