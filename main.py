import argparse
import os
import torch
import torchvision

import models
from utils import progress_bar, Chrono, Logger

log_msg = '{}, {:.2f}, {:.10f}, {:.6f}, {:.4f}, {:.6f}, {:.4f}\n'


class Cifar10:
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
        self.saveFile = '%s_%s' % (args.model, args.experiment)

        self.chrono = Chrono()

        if not os.path.isdir(args.log_path):
            os.makedirs(args.log_path)

        self.logger = Logger('%s/%s-%s.csv' % (args.log_path, args.model, args.experiment))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Going to run on %s' % self.device)

        self.trainloader, self.testloader = self.dataloader()

        print('==> Building model..')
        self.model = getattr(models, args.model)()

        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=self.lr_schedule,
                                                                   last_epoch=-1)

        if args.resume:
            self.load()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        self.model = self.model.to(self.device)

    def lr_schedule(self, epoch):
        return 0.1 if epoch % 30 == 0 else 1

    def run(self):
        if self.test_only:
            self.test()
        else:
            for epoch in range(self.epoch, self.max_epoch):
                print('\nEpoch: %d' % (epoch + 1))
                self.lr = self.optimizer.param_groups[0]['lr']

                with self.chrono.measure("epoch"):
                    self.train()
                    self.test()
                    self.scheduler.step()

                self.epoch = epoch
                self.log()

                if self.test_acc > self.best_acc:
                    self.save()

    def train(self):
        self.model.train()
        self.train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            with self.chrono.measure("step_time"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                self.train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            progress_bar(self.chrono, batch_idx, len(self.trainloader), 'Lr: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (self.lr, self.train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.chrono.remove("step_time")
        self.train_acc = 100. * correct / total

    def test(self):
        self.model.eval()
        self.test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                with self.chrono.measure("step_time"):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    self.test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                progress_bar(self.chrono, batch_idx, len(self.testloader), 'Lr: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (self.lr, self.test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.chrono.remove("step_time")
        self.test_acc = 100. * correct / total
        self.last_test_loss = self.test_loss

    def dataloader(self):
        temp_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        temp_set = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=temp_transform)
        mean = temp_set.data.mean(axis=(0, 1, 2)) / 255
        std = temp_set.data.std(axis=(0, 1, 2)) / 255

        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=12, shuffle=False)

        return train_dataloader, test_dataloader

    def load(self):
        print('==> Loading from save...')
        assert os.path.isdir('./state_dicts'), 'Error: no state_dicts directory found!'
        state_dict = torch.load('./state_dicts/%s.pth' % self.saveFile, map_location='cpu')
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']
        self.best_acc = state_dict['acc']
        self.lr = self.optimizer.param_groups[0]['lr']
        if not self.test_only:
            print('%s epoch(s) will run, save already has %s epoch(s) and best %s accuracy'
                  % ((self.max_epoch - self.epoch), self.epoch, self.best_acc))

    def save(self):
        print('Saving..')
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'acc': self.best_acc,
            'epoch': self.epoch
        }
        if not os.path.isdir('state_dicts'):
            os.mkdir('state_dicts')
        torch.save(state, './state_dicts/%s.pth' % self.saveFile)
        self.best_acc = self.test_acc

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
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-e', '--epoch', default=200, help='Epoch count to run in total')
    parser.add_argument('-x', '--experiment', default=1, help='Experiment number')
    parser.add_argument('-lp', '--log_path', default='logs', help='Path that log files stored')
    parser.add_argument('-m', '--model', required=True, choices=list(Cifar10.models), help='Model to run')
Cifar10(parser.parse_args()).run()
