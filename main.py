import argparse
import os

import numpy
import torch

import models
from models.auto_encoder import auto_encoder as AutoEncoder
from utils import ProgressBar, Chrono, Logger, Utils, dataloader, update_lr, get_torch_vars

log_msg = '{}, {:.2f}, {:.10f}, {:.6f}, {:.4f}, {:.6f}, {:.4f}\n'


class Cifar10:
    epochs = [1, 5, 10, 15, 20]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    models = ('bit', 'resnet')
    epoch = 0

    train_acc = 0
    test_acc = 0
    best_acc = 0

    train_loss = 0
    test_loss = 0
    last_test_loss = 0

    def __init__(self, args):
        self.initial_lr = args.learning_rate
        self.lr = args.learning_rate
        self.test_only = args.test_only
        self.max_epoch = args.epoch
        self.saveFile = '%s' % args.model
        self.experiment = '%s' % args.experiment

        self.progress_bar = ProgressBar()
        self.chrono = Chrono()

        if not os.path.isdir(args.log_path):
            os.makedirs(args.log_path)

        self.logger = Logger('%s/%s-%s.csv' % (args.log_path, args.model, args.experiment))

        self.trainset, self.testset, self.trainloader, self.testloader = dataloader()

        print('==> Building model..')
        self.ae = AutoEncoder()
        self.model = getattr(models, args.model)()

        if args.model == 'bit':
            self.model.load_from(numpy.load('./state_dicts/%s.npz' % self.saveFile))

        if torch.cuda.is_available():
            self.ae = torch.nn.DataParallel(self.ae)
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        self.load_ae()
        if args.resume:
            self.load()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = get_torch_vars(self.criterion, False)

        self.ae = get_torch_vars(self.ae, False)
        self.model = get_torch_vars(self.model, False)

    def run(self):
        if self.test_only:
            self.test()
        else:
            for epoch in range(self.epoch, self.max_epoch):
                self.epoch = epoch
                print('\nEpoch: %d' % (self.epoch + 1))

                with self.chrono.measure("epoch"):
                    self.train()
                    self.test()

                self.log()

                if self.test_acc > self.best_acc:
                    self.save()

    def train(self):
        self.ae.train()
        self.model.train()
        self.train_loss = 0
        correct = 0
        total = 0
        self.progress_bar.newbar(len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            with self.chrono.measure("step_time"):
                inputs = get_torch_vars(inputs)
                targets = get_torch_vars(targets)

                self.lr = update_lr(self.optimizer,
                                    self.epoch, self.epochs,
                                    self.initial_lr,
                                    batch_idx, len(self.trainloader))
                if self.lr is None:
                    break

                self.optimizer.zero_grad()
                encoded, _ = self.ae(inputs)
                outputs = self.model(encoded)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                self.train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            msg = 'Step: %s | Tot: %s | Lr: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
                  (Utils.format_time(self.chrono.last('step_time')),
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
        self.ae.eval()
        self.model.eval()
        self.test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            self.progress_bar.newbar(len(self.testloader))
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                with self.chrono.measure("step_time"):
                    inputs = get_torch_vars(inputs)
                    targets = get_torch_vars(targets)
                    encoded, _ = self.ae(inputs)
                    outputs = self.model(encoded)
                    loss = self.criterion(outputs, targets)

                    self.test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                msg = 'Step: %s | Tot: %s | Lr: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
                      (Utils.format_time(self.chrono.last('step_time')),
                       Utils.format_time(self.chrono.total('step_time')),
                       self.lr,
                       self.test_loss / (batch_idx + 1),
                       100. * correct / total,
                       correct,
                       total)
                self.progress_bar.update(batch_idx, msg)

        self.chrono.remove("step_time")
        self.test_acc = 100. * correct / total
        self.last_test_loss = self.test_loss

    def load(self):
        print('==> Loading from save...')
        assert os.path.isdir('./state_dicts'), 'Error: no state_dicts directory found!'
        state_dict = torch.load('./state_dicts/%s_%s.pth' % (self.saveFile, self.experiment), map_location='cpu')
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch'] + 1
        self.best_acc = state_dict['acc']
        if not self.test_only:
            print('%s epoch(s) will run, save already has %s epoch(s) and best %s accuracy'
                  % ((self.max_epoch - self.epoch), self.epoch, self.best_acc))

    def load_ae(self):
        self.ae.load_state_dict(torch.load('./state_dicts/autoencoder.pkl', map_location='cpu'))

    def save(self):
        self.best_acc = self.test_acc
        print('Saving..')
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'acc': self.best_acc,
            'epoch': self.epoch
        }
        if not os.path.isdir('state_dicts'):
            os.mkdir('state_dicts')
        torch.save(state, './state_dicts/%s_%s.pth' % (self.saveFile, self.experiment))

    def log(self):
        self.logger.write(log_msg.format(self.epoch + 1,
                                         self.chrono.last("epoch"),
                                         self.lr,
                                         self.train_loss / len(self.trainloader), self.train_acc,
                                         self.test_loss / len(self.testloader), self.test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('-r', '--resume', action='store_true', help='resume from save')
    parser.add_argument('-t', '--test_only', action='store_true', help='Test only')
    parser.add_argument('-l', '--learning_rate', default=3e-3, type=float, help='learning rate')
    parser.add_argument('-e', '--epoch', default=200, help='Epoch count to run in total')
    parser.add_argument('-x', '--experiment', default=1, help='Experiment number')
    parser.add_argument('-lp', '--log_path', default='logs', help='Path that log files stored')
    parser.add_argument('-m', '--model', required=True, choices=list(Cifar10.models), help='Model to run')
    Cifar10(parser.parse_args()).run()
