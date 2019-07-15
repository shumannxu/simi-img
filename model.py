import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class BookModel(nn.Module):
    def __init__(self, n_classes=1):
        super(BookModel, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(True),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, 5, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(3, stride=2),
        )
        self.n_classes = n_classes
        self.cls = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.Dropout(0.05),
            nn.ReLU(True),
            #
            # nn.Linear(4096, 4096),
            # nn.Dropout(0.05),
            # nn.Sigmoid(),

            nn.Linear(4096, self.n_classes),
            nn.Sigmoid()
        )

    def forward_features(self, input2, arr):
        batch = input2.size(0)
        feature2 = self.cov(input2)
        b = feature2.view(batch, 9216)
        arr.append(b.detach().cpu().numpy())
        return arr

    def forward1(self, input1):
        batch = input1.size(0)
        feature1 = self.cov(input1)
        a = feature1.view(batch, 9216)
        print('load features')
        features = np.load('./features.npy')
        scores = []
        for feature in features:
            feature = torch.from_numpy(feature).cuda()
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
        a = feature1.view(batch, 9216)
        b = feature2.view(batch, 9216)
        pdist = torch.abs(torch.sub(a,b))

        weights = F.softmax(pdist)
        pdist_att = torch.mul(pdist, weights)
        return self.cls(pdist)



    def forward(self, train = False, extract_features=False, input2=None, arr=None, input1=None):
        if train:
            return self.forward_train(input1, input2)
        elif extract_features:
            return self.forward_features(input2, arr)
        else:
            return self.forward1(input1)


if __name__ == '__main__':
    m = BookModel()
    x1 = torch.ones(1,3, 224, 224)
    x2 = torch.ones(1,3, 224, 224)
    print(m(x1,x2))
