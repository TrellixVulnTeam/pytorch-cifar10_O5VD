from utils import Logger


def learning_scheduler(lr):
    if epoch < epochs[0]:
        lr = lr * (step + 1 + total_step * epoch) / (total_step * epochs[0])
    elif epoch >= epochs[-1]:
        lr = None
    else:
        for s in epochs[1:]:
            if s < epoch:
                lr /= 10
    return lr


lr = 0.003
epochs = [1, 5, 10, 15, 20]
total_step = 391
logger = Logger('logs/lr.csv')
logger.header("epoch, step, lr")

for epoch in range(0, epochs[-1]):
    for step in range(0, total_step):
        logger.write('{}, {}, {:.10f}\n'.format(epoch + 1,
                                                step + 1,
                                                learning_scheduler(lr)))
