import argparse
import os
import torch
import torchvision

import models
from utils import progress_bar


class Cifar10:
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    models = ('alexnet', 'bit', 'densenet', 'googlenet', 'resnet', 'vgg')
    epoch = 0
    best_acc = 0
    acc = 0

    # noinspection PyInitNewSignature
    def __init__(self, args):
        self.lr = args.learning_rate
        self.test_only = args.test_only
        self.max_epoch = args.epoch
        self.saveFile = '%s.pth' % args.model

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Going to run on %s' % self.device)

        self.trainloader, self.testloader = self.dataloader()

        self.model = self.build_model(args.model)

        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        if args.resume:
            self.load()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        self.model = self.model.to(self.device)

    def run(self):
        if self.test_only:
            self.test()
        else:
            for epoch in range(self.epoch, self.max_epoch):
                print('\nEpoch: %d' % epoch)
                self.epoch = epoch
                self.train()
                self.test()

                if self.acc > self.best_acc:
                    self.save()

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        lr = self.adjust_learning_rate()
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Lr: %.5f'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total, lr))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.acc = 100. * correct / total

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1 ** (self.epoch // 15))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

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

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=4, shuffle=False)

        return train_dataloader, test_dataloader

    def build_model(self, model):
        print('==> Building model..')
        return getattr(models, model)()

    def load(self):
        print('==> Loading from save...')
        assert os.path.isdir('./state_dicts'), 'Error: no state_dicts directory found!'
        state_dict = torch.load('./state_dicts/%s' % self.saveFile, map_location='cpu')
        if 'model' in state_dict:
            self.model.load_state_dict(state_dict['model'])
            self.epoch = state_dict['epoch']
            self.best_acc = state_dict['acc']
        else:
            if 'module' in state_dict:
                self.model.load_state_dict(state_dict)
            else:
                self.model.module.load_state_dict(state_dict)
        if not self.test_only:
            print('%s epoch(s) will run, save already has %s epoch(s) and best %s accuracy'
                  % ((self.max_epoch - self.epoch), self.epoch, self.best_acc))

    def save(self):
        print('Saving..')
        state = {
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'acc': self.acc,
            'epoch': self.epoch,
        }
        if not os.path.isdir('state_dicts'):
            os.mkdir('state_dicts')
        torch.save(state, './state_dicts/%s' % self.saveFile)
        self.best_acc = self.acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('-r', '--resume', action='store_true', help='resume from save')
    parser.add_argument('-t', '--test_only', action='store_true', help='Test only')
    parser.add_argument('-l', '--learning_rate', default=0.1, type=float, help='learning rate')
    parser.add_argument('-e', '--epoch', default=200, type=float, help='Epoch count to run in total')
    parser.add_argument('-m', '--model', required=True, choices=list(Cifar10.models), help='Model to run')
    Cifar10(parser.parse_args()).run()
