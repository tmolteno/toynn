#
#   Minimal PyTorch example.
#
#   Licensed under GPLv3. Copyright (c) 2020. Max Scheel, max@max.ac.nz
#
import os
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(1, '../')
from gen_data import gen_dataset, plot_data

class Net(nn.Module):
    def __init__(self, SIGNAL_LENGTH, NUM_TYPES):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(SIGNAL_LENGTH, 125)
        self.fc2 = nn.Linear(125, 64)
        self.fc3 = nn.Linear(64, NUM_TYPES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':

    SIGNAL_LENGTH = 512
    TRAINING_N = 150000
    TEST_N = 10000
    NUM_TYPES = 3
    np.random.seed(101)

    PATH = './model_{}_{}.pth'.format(SIGNAL_LENGTH, NUM_TYPES)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data, labels, t = gen_dataset(TEST_N + TRAINING_N, SIGNAL_LENGTH, NUM_TYPES)

    if not (os.path.exists(PATH)):
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(data[TEST_N:]),
            torch.Tensor(labels[TEST_N:]).long())
        trainloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=256)

        model = Net(SIGNAL_LENGTH, NUM_TYPES)
        model.to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adamax(model.parameters())

        for epoch in range(12):
            running_loss = 0.0
            for i, databatch in enumerate(trainloader):
                inputs, target_labels = databatch[0].to(
                    device), databatch[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, target_labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[{}, {}] loss {}'.format(epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
        torch.save(model.state_dict(), PATH)
    else:
        model = Net(SIGNAL_LENGTH, NUM_TYPES)
        model.load_state_dict(torch.load(PATH))

    model.to('cpu')
    test_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(data[0:TEST_N]), torch.Tensor(labels[0:TEST_N]).long())
    testloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=256)

    correct = 0
    total = 0
    with torch.no_grad():
        for databatch in testloader:
            inputs, true_labels = databatch
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()

    print('Accuracy on {} test: {} %'.format(TEST_N, 100 * correct / total))

    testloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=4)
    with torch.no_grad():
        for (i, databatch) in enumerate(testloader):
            if i < 10:
                inputs, true_labels = databatch
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                print(i, outputs.data, predicted, true_labels)
