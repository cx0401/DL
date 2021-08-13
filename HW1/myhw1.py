import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class myData(Dataset):
    def __init__(self, file):
        with open(file, "r") as f:
            data = list(csv.reader(f))
            data = np.array(data[1:])[:, 1:].astype(float)
            data = torch.FloatTensor(data)
        # data = np.loadtxt(file)
        self.x = data[:, : -1]
        self.y = data[:, [-1]]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(93, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = myModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0)


def trainNet():
    dataloader = DataLoader(myData("data/covid.train.csv"), batch_size=4, shuffle=True)
    for epoch in range(5000):
        model.train()
        for x, y in dataloader:
            x, y = x.to(torch.float32), y.to(torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            # if epoch % 1000 == 0:
            #     print(x, model(x), y)


def validation():
    dataloader = DataLoader(myData("data/covid.test.shuffle.csv"), batch_size=1, shuffle=False)
    model.eval()
    loss = 0
    for x, y in dataloader:
        with torch.no_grad():
            x, y = x.to(torch.float32), y.to(torch.float32)
            loss += criterion(model(x), y).cpu().item() * len(x)
            print(model(y), x)
    print(loss / len(dataloader))


trainNet()
validation()
