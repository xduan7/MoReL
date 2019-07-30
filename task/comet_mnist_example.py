""" 
    File Name:          MoReL/comet_mnist_example.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               7/25/19
    Python Version:     3.5.4
    File Description:   

"""
from comet_ml import Optimizer
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F


# We only need to specify the algorithm and hyperparameters to use:
config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare your hyperparameters in the Vizier-inspired format:
    "parameters": {
        "batch_size": {"type": "discrete", 'values': [32, 64, 128, 256, 512]},
        "num_epochs": 1,
        "learning_rate": 0.001

    },

    # Declare what we will be optimizing, and how:
    "spec": {
        "metric": "train_loss",
        "objective": "minimize",
        # "batch_size": 100,
    },
}

optimizer = Optimizer(config, project_name="MNIST")

# hyper_params = {
#     "sequence_length": 28,
#     "input_size": 28,
#     "hidden_size": 128,
#     "num_layers": 2,
#     "num_classes": 10,
#     "batch_size": 100,
#     "num_epochs": 10,
#     "learning_rate": 0.01
# }


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# experiment = Experiment(project_name="pytorch")
for experiment in optimizer.get_experiments():

    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=experiment.get_parameter('batch_size'),
        shuffle=True,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=experiment.get_parameter('batch_size'),
        shuffle=False,
        pin_memory=True)

    model = Net().to('cuda')

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment.get_parameter('learning_rate'))

    # Train the Model
    with experiment.train():
        for epoch in range(experiment.get_parameter('num_epochs')):
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(train_loader):

                images = images.view(-1, 1, 28, 28)
                images, labels = images.to('cuda'), labels.to('cuda')

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Compute train accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum()

                experiment.log_metric('loss', loss.item())

                if (i + 1) % 100 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                          % (epoch + 1, experiment.get_parameter(
                        'num_epochs'), i + 1, len(train_dataset) //
                             experiment.get_parameter('batch_size'),
                             loss.item()))

    with experiment.test():
        # Test the Model
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 1, 28, 28)
            images, labels = images.to('cuda'), labels.to('cuda')

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        experiment.log_metric("accuracy", 100. * correct / total)
        print('Test Accuracy of the model on the 10000 test images: %d %%' %
              (100. * correct / total))\

