def accuracy(targets, preds, batch_size):
    correct = sum(targets == preds).cpu()
    acc = (correct/batch_size * 100)
    return acc