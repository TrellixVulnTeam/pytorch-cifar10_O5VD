import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from models.auto_encoder import auto_encoder as AutoEncoder
from utils import Utils, dataloader, update_lr, get_torch_vars, imshow, Chrono, ProgressBar


def valid():
    print("Loading checkpoint...")
    autoencoder.load_state_dict(torch.load("./state_dicts/autoencoder.pkl"))
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images = get_torch_vars(images)
    decoded_imgs = autoencoder(images)[1]

    imshow(torchvision.utils.make_grid(images))
    imshow(torchvision.utils.make_grid(decoded_imgs.data))


def train():
    autoencoder.train()
    epochs = [1, 5, 10]
    for epoch in range(epochs[-1]):
        running_loss = 0.0
        progress_bar.newbar(len(trainloader))
        for batch_idx, (inputs, _) in enumerate(trainloader):
            with chrono.measure("step_time"):
                inputs = get_torch_vars(inputs)

                lr = update_lr(optimizer, epoch, epochs, 0.001, batch_idx, len(trainloader))
                if lr is None:
                    break

                encoded, outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.data

            msg = 'Step: %s | Tot: %s | LR: %.10f | Loss: %.3f' % \
                  (Utils.format_time(chrono.last('step_time')),
                   Utils.format_time(chrono.total('step_time')),
                   lr,
                   running_loss / (batch_idx + 1))
            progress_bar.update(batch_idx, msg)

        chrono.remove("step_time")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    SEED = 87
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    chrono = Chrono()
    progress_bar = ProgressBar()

    _, _, trainloader, testloader = dataloader()

    autoencoder = get_torch_vars(AutoEncoder(), False)
    if args.valid:
        valid()
        exit(0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(autoencoder.parameters())

    train()

    print('Finished Training')
    print('Saving Model...')
    if not os.path.exists('./state_dicts'):
        os.mkdir('./state_dicts')
    torch.save(autoencoder.state_dict(), "./state_dicts/autoencoder.pkl")
