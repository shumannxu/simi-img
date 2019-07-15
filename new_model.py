import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class BookModel2(nn.Module):
    def __init__(self, n_class=1):
        super(BookModel2, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv2d(3, 64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(2),

        )
        self.n_class = n_class

        self.cls = nn.Sequential(
            nn.Linear(6400, 2048),
            nn.ReLU(True),
            nn.Linear(2048, n_class),
            nn.Sigmoid()
        )


    def forward_features(self, input2, arr):
        batch = input2.size(0)
        feature2 = self.cov(input2)
        b = feature2.view(batch, 6400)
        arr.append(b.detach().numpy())
        return arr

    def forward1(self, input1):
        batch = input1.size(0)
        feature1 = self.cov(input1)
        a = feature1.view(batch, 6400)
        print('load features')
        features = np.load('./features.npy')
        scores = []
        for feature in features:
            feature = torch.from_numpy(feature)
            pdist  = torch.abs(torch.sub(a,feature))
            weights = F.softmax(pdist)
            pdist_att = torch.mul(pdist, weights)
            score = self.cls(pdist_att)
            scores.append(score)
        return np.argsort(scores)[-5:][::-1]


    def forward_train(self, input1, input2):
        batch = input1.size(0)
        feature1 = self.cov(input1)
        feature2 = self.cov(input2)
        a = feature1.view(batch, 6400)
        b = feature2.view(batch, 6400)
        pdist = torch.abs(torch.sub(a,b))

        weights = F.softmax(pdist)
        pdist_att = torch.mul(pdist, weights)
        return self.cls(pdist_att)



    def forward(self, train = False, extract_features=False, input2=None, arr=None, input1=None):
        if train:
            return self.forward_train(input1, input2)
        elif extract_features:
            return self.forward_features(input2, arr)
        else:
            return self.forward1(input1)


if __name__ == '__main__':
    m = BookModel2()
    x1 = torch.ones(1,3, 224, 224)
    x2 = torch.ones(1,3, 224, 224)
    print(m(train = True,input1 =x1,input2 =x2))
