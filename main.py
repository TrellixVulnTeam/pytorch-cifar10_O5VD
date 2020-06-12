import argparse
import os
import sys

import numpy
import torch

import models
from utils import ProgressBar, Chrono, Logger, Utils, dataloader, get_torch_vars


class Cifar10:
    log_msg = '{}, {:.2f}, {:.10f}, {:.6f}, {:.4f}, {:.6f}, {:.4f}\n'
    step_msg = 'Step: %s | Tot: %s | Lr: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)'

    epochs = [1, 5, 10, 15, 20]
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    models = ('bit', 'resnet')
    epoch = 0

    train_acc = 0
    test_acc = 0
    best_acc = 0

    train_loss = 0
    test_loss = 0

    confusion_matrix = None

    def __init__(self, args):
        self.initial_lr = args.learning_rate
        self.lr = args.learning_rate
        self.test_only = args.test_only
        self.dump_statistics = args.dump_statistics
        self.modelName = args.model
        self.experiment = args.experiment
        self.log_path = args.log_path
        self.save_path = args.save_path

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        self.logger = Logger('%s/%s_%s.csv' % (self.log_path, self.modelName, args.experiment),
                             'epoch, time, learning_rate, tr_loss, tr_acc, val_loss, val_acc')
        self.progress_bar = ProgressBar()
        self.chrono = Chrono()

        self.trainset, self.testset, self.trainloader, self.testloader = dataloader()

        print('==> Building model..')
        self.model = getattr(models, self.modelName)()

        if self.modelName == 'bit':
            self.model.load_from(numpy.load('./state_dicts/%s.npz' % self.modelName))

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        if args.resume or self.test_only or self.dump_statistics:
            self.load()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = get_torch_vars(self.criterion, False)

        self.model = get_torch_vars(self.model, False)

    def run(self):
        if self.test_only:
            self.test()
        elif self.dump_statistics:
            self.test()
            self.dump_cmx()
            self.dump_measurements()
        else:
            for epoch in range(self.epoch, self.epochs[-1]):
                self.epoch = epoch
                print('\nEpoch: %d' % (self.epoch + 1))

                with self.chrono.measure("epoch"):
                    self.train()
                    self.test()

                self.log()

                if self.test_acc > self.best_acc:
                    self.save()

    def train(self):
        self.model.train()
        self.train_loss = 0
        correct = 0
        total = 0
        self.progress_bar.newbar(len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            with self.chrono.measure("step_time"):
                inputs = get_torch_vars(inputs)
                targets = get_torch_vars(targets)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                self.train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            msg = self.step_msg % (Utils.format_time(self.chrono.last('step_time')),
                                   Utils.format_time(self.chrono.total('step_time')),
                                   self.lr,
                                   self.train_loss / (batch_idx + 1),
                                   100. * correct / total,
                                   correct,
                                   total)
            self.progress_bar.update(batch_idx, msg)

        self.chrono.remove("step_time")
        self.train_acc = 100. * correct / total

    def test(self):
        self.model.eval()
        self.test_loss = 0
        correct = 0
        total = 0
        self.confusion_matrix = torch.zeros([len(self.classes), len(self.classes)], dtype=torch.int)
        with torch.no_grad():
            self.progress_bar.newbar(len(self.testloader))
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                with self.chrono.measure("step_time"):
                    inputs = get_torch_vars(inputs)
                    targets = get_torch_vars(targets)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    self.test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    for t, p in zip(targets.view(-1), predicted.view(-1)):
                        self.confusion_matrix[t.long(), p.long()] += 1

                msg = self.step_msg % (Utils.format_time(self.chrono.last('step_time')),
                                       Utils.format_time(self.chrono.total('step_time')),
                                       self.lr,
                                       self.test_loss / (batch_idx + 1),
                                       100. * correct / total,
                                       correct,
                                       total)
                self.progress_bar.update(batch_idx, msg)

        self.chrono.remove("step_time")
        self.test_acc = 100. * correct / total

    def load(self):
        print('==> Loading from save...')
        assert os.path.isdir('./%s' % self.save_path), 'Error: save directory not found!'
        state_dict = torch.load('./%s/%s_%s.pth' % (self.save_path, self.modelName, self.experiment))
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch'] + 1
        self.best_acc = state_dict['acc']
        print('%s epoch(s) will run, save already has %s epoch(s) and best %s accuracy'
              % ((self.epochs[-1] - self.epoch), self.epoch, self.best_acc))

    def save(self):
        self.best_acc = self.test_acc
        print('Saving..')
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'acc': self.best_acc,
            'epoch': self.epoch
        }
        if not os.path.isdir('./%s' % self.save_path):
            os.mkdir('./%s' % self.save_path)
        torch.save(state, './%s/%s_%s.pth' % (self.save_path, self.modelName, self.experiment))

        self.dump_cmx()
        self.dump_measurements()

    def dump_cmx(self):
        with open('./%s/%s_%s.cmx' % (self.save_path, self.modelName, self.experiment), 'w') as f:
            print(self.confusion_matrix.cpu().data.numpy(), file=f)

    def log(self):
        self.logger.write(self.log_msg.format(self.epoch + 1,
                                              self.chrono.last("epoch"),
                                              self.lr,
                                              self.train_loss / len(self.trainloader), self.train_acc,
                                              self.test_loss / len(self.testloader), self.test_acc))

    def dump_measurements(self):
        with open('./%s/%s_%s.mea' % (self.save_path, self.modelName, self.experiment), 'w') as f:
            tp = self.confusion_matrix.diag()
            for c in range(len(self.classes)):
                idx = torch.ones(len(self.classes)).byte()
                idx[c] = 0

                tn = self.confusion_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
                fp = self.confusion_matrix[idx, c].sum()
                fn = self.confusion_matrix[c, idx].sum()

                print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c, tp[c], tn, fp, fn), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('-r', '--resume', action='store_true', help='resume from save')
    parser.add_argument('-t', '--test_only', action='store_true', help='Test only')
    parser.add_argument('-s', '--dump_statistics', action='store_true', help='Test and save all statistics')
    parser.add_argument('-l', '--learning_rate', default=1e-2, type=float, help='learning rate')
    parser.add_argument('-x', '--experiment', default=1, help='Experiment number')
    parser.add_argument('-m', '--model', required=True, choices=list(Cifar10.models), help='Model to run')
    parser.add_argument('-lp', '--log_path', default='logs', help='Path that log files stored')
    parser.add_argument('-sp', '--save_path', default='state_dicts', help='Path that pytorch save files stored')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
    Cifar10(parser.parse_args()).run()
