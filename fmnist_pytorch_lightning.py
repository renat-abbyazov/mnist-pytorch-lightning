import logging

import torch
from torchvision import datasets, transforms
import random
import numpy as np
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F


seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

class LitFMNIST(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        self.trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=False,
                                         train=True, transform=transform)
        self.testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=False,
                                             train=False,
                                        transform=transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=False,
                                                  worker_init_fn=random.seed(seed))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return {'loss': loss}

    def val_dataloader(self):
        # testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=False,
                                                 worker_init_fn=random.seed(seed))
        return testloader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # logits = self(x)
        logits = self(x)
        loss = F.nll_loss(logits, y)

        pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        val_acc = pred.eq(y.view_as(pred)).sum().to(dtype=torch.float)
        # pred = torch.argmax(y_hat, dim=1)
        # val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        # val_acc = torch.tensor(val_acc)
        # print(val_acc)
        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['val_acc'] for x in outputs]) / len(self.val_dataloader().dataset)
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).sum() / len(self.val_dataloader().dataset)

        # avg_loss = outputs['val_loss'].mean()
        tensorboard_logs = {'val_loss': avg_loss}
        # tensorboard_logs = {'val_loss': avg_acc}
        # logger = logging.getLogger(__name__)
        # logger.info(avg_acc)
        print(avg_acc)
        # self.logger.info(avg_acc)
        # print(avg_acc)
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.003)



model = LitFMNIST()

# trainer = pl.Trainer(max_steps=10)
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model)

# # Define a transform to normalize the data
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))])
# # Download and load the training data
# trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, worker_init_fn=random.seed(seed))
#
# # Download and load the test data
# testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
# # testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, worker_init_fn=random.seed(seed))
#
#
#
#
# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         # make sure input tensor is flattened
#         x = x.view(x.shape[0], -1)
#
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.log_softmax(self.fc4(x), dim=1)
#         return x
#
#
# def debug_shapes_and_accuracy():
#     model = Classifier()
#
#     images, labels = next(iter(testloader))
#     # Get the class probabilities
#     ps = torch.exp(model(images))
#     # Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
#     # print(ps.shape)
#     top_p, top_class = ps.topk(1, dim=1)
#     # Look at the most likely classes for the first 10 examples
#     # print(top_class[:2, :])
#     equals = top_class == labels.view(*top_class.shape)
#     accuracy = torch.mean(equals.type(torch.FloatTensor))
#
#     print(f'Accuracy: {accuracy.item() * 100}%')
#
#     logits = model(images)
#     top_p, top_class = logits.topk(1, dim=1)
#     equals = top_class == labels.view(*top_class.shape)
#     accuracy = torch.mean(equals.type(torch.FloatTensor))
#     print(f'Accuracy: {accuracy.item() * 100}%')
#
#     # different ways to calc accuracy
#     equals1 = top_class == labels.view(*top_class.shape)
#     equals2 = top_class.view(*labels.shape) == labels
#     import numpy as np
#
#     acc1 = np.mean(equals1.detach().cpu().numpy())
#     acc2 = np.mean(equals2.detach().cpu().numpy())
#
#     from sklearn.metrics import accuracy_score
#     acc3 = accuracy_score(top_class.view(*labels.shape).detach().cpu().numpy(),
#                           labels.detach().cpu().numpy())
#     print("accuracy variant 1", acc1)
#     print("accuracy variant 2", acc2)
#     print("accuracy variant 3", acc3)
#
# debug_shapes_and_accuracy()
#
# model = Classifier()
# criterion = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.003)
# # optimizer = optim.Adam(model.parameters(), lr=0.003)
#
# epochs = 1
# steps = 0
#
# train_losses, test_losses = [], []
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in trainloader:
#
#         optimizer.zero_grad()
#
#         log_ps = model(images)
#         loss = criterion(log_ps, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     else:
#         test_loss = 0
#         test_loss2 = 0
#         accuracy = 0
#         accuracy2 = 0
#         accuracy3 = 0
#         correct = 0
#
#         # Turn off gradients for validation, saves memory and computations
#         with torch.no_grad():
#             for images, labels in testloader:
#                 log_ps = model(images)
#                 test_loss += criterion(log_ps, labels)
#
#                 ps = torch.exp(log_ps)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == labels.view(*top_class.shape)
#                 accuracy += torch.mean(equals.type(torch.FloatTensor))
#
#                 test_loss2 += F.nll_loss(log_ps, labels)
#                 ps = torch.exp(log_ps)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == labels.view(*top_class.shape)
#                 accuracy2 += torch.mean(equals.type(torch.FloatTensor))
#
#                 ps = torch.exp(log_ps)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == labels.view(*top_class.shape)
#                 accuracy3 += torch.sum(equals.type(torch.FloatTensor))
#
#                 output = model(images)
#                 target = labels
#                 pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#                 correct += pred.eq(target.view_as(pred)).sum().item()
#
#         train_losses.append(running_loss / len(trainloader))
#         test_losses.append(test_loss / len(testloader))
#
#         print("Epoch: {}/{}.. ".format(e + 1, epochs),
#               "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
#               "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
#               "Test Loss2: {:.3f}.. ".format(test_loss2 / len(testloader)),
#               "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
#               "Test Accuracy2: {:.3f}".format(accuracy2 / len(testloader)),
#               "Test Accuracy3: {:.3f}".format(accuracy3 / len(testloader.dataset)),
#               "Test Accuracy4: {:.3f}".format(correct / len(testloader.dataset)),
#               )
# # Epoch: 1/1..  Training Loss: 2.131..  Test Loss: 1.748..  Test Loss2: 1.748..  Test Accuracy: 0.400
