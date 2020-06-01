import argparse
import os
import torch
import torchvision

import models
from utils import progress_bar


class Cifar10:
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    models = (
        'alexnet',
        'densenet121', 'densenet161', 'densenet169', 'densenet201',
        'googlenet',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
    )
    epoch = 0
    best_acc = 0
    acc = 0

    def __init__(self, args):
        self.test_only = args.test_only
        self.max_epoch = args.epoch
        self.saveFile = '%s.pth' % args.model

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.trainloader = self.train_dataloader()
        self.testloader = self.test_dataloader()

        self.model = self.build_model(args.model)
        self.net = self.model.to(self.device)

        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            torch.backends.cudnn.benchmark = True

        if args.resume:
            self.load()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    def run(self):
        if self.test_only:
            self.test()
        else:
            for epoch in range(self.epoch, self.max_epoch):
                print('\nEpoch: %d' % epoch)
                self.train()
                self.test()

                if self.acc > self.best_acc:
                    self.save()

    def train(self):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.acc = 100. * correct / total

    def train_dataloader(self):
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                          torchvision.transforms.RandomHorizontalFlip(),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                           (0.2023, 0.1994, 0.2010))])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)
        return dataloader

    def test_dataloader(self):
        transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                         (0.2023, 0.1994, 0.2010))])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_val)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=4, shuffle=False)
        return dataloader

    def build_model(self, model):
        print('==> Building model..')
        return getattr(models, model)()

    def load(self):
        print('==> Loading from save...')
        assert os.path.isdir('./state_dicts'), 'Error: no state_dicts directory found!'
        state_dict = torch.load('./state_dicts/%s' % self.saveFile)
        if 'net' in state_dict:
            if 'module' in state_dict['net']:
                self.net.load_state_dict(state_dict['net'])
            else:
                self.net.module.load_state_dict(state_dict['net'])
            self.epoch = state_dict['epoch']
            self.best_acc = state_dict['acc']
        else:
            if 'module' in state_dict:
                self.net.load_state_dict(state_dict)
            else:
                self.net.module.load_state_dict(state_dict)
        if not self.test_only:
            print('%s epoch(s) will run, save already has %s epoch(s)' % ((self.max_epoch - self.epoch), self.epoch))

    def save(self):
        print('Saving..')
        state = {
            'net': self.net.state_dict(),
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
    parser.add_argument('-m', '--model', required=True, help='Model to run ')  # TODO
    args = parser.parse_args()
    Cifar10(args).run()
