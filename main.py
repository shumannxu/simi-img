import torch
import torchvision
import torch.nn as nn
import os
import random
from model import BookModel
from dataset import BookDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision
train_set = '../dataset/train/'
val_set = '../dataset/test/'

test_image = [os.listdir(val_set)[0]]
train_image = os.listdir(train_set)

momentum = 0.9
lr = 0.001
epoch = 20
batch_size = 3


def max_pool(feature):
    b, c, w, h = feature.size(0), feature.size(1), feature.size(2), feature.size(3)
    feature = feature.detach().numpy()
    new_feature = np.zeros((b, c))
    for i in range(c):
        new_feature[0][i] = np.max(feature[0][i, :, :])
    return new_feature[0]


def train():
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((299, 299)),
        torchvision.transforms.ToTensor()
    ])
    model_use = 'vgg'
    # criterion = torch.nn.BCELoss()
    # model = BookModel()
    model = {}
    vgg_model = torchvision.models.vgg16_bn(pretrained=True)
    incep_model = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
    model['vgg'] = vgg_model
    model['incep'] = incep_model
    """
    512 * 7 * 7
    """

    # vgg_model = torch.nn.Sequential(
    #     torch.nn.MaxPool2d(7)
    # )
    model = model[model_use]

    if not os.path.exists('./'+model_use+'_features.npy'):
        print('extract features...')
        all_feature = []
        for img in tqdm(train_image):
            input = test_transform(Image.open(os.path.join(train_set, img)))
            feature = model(input.unsqueeze(dim=0))
            # if model_use == 'vgg':
            #     all_feature.append(max_pool(feature))
            # else:
            all_feature.append(feature)
        np.save('./'+model_use+'_features.npy', all_feature)
    else:
        print('ok')
        all_feature = np.load('./'+model_use+'_features.npy')

    query_index = random.randint(0, len(all_feature)-1)
    query_feature = all_feature[query_index]

    euclu = torch.nn.PairwiseDistance(p=2)
    dists = [euclu(torch.from_numpy(query_feature).unsqueeze(dim=0), torch.from_numpy(feature).unsqueeze(dim=0)) for feature in all_feature]

    sorted_index = np.argsort(dists)[:5]
    print(sorted_index)
    plt.figure(figsize=(20,4))
    plt.subplot(1, 6, 1)
    plt.imshow(Image.open(os.path.join(train_set, train_image[query_index])))

    for i, sidx in enumerate(sorted_index):
        plt.subplot(1, 6, i+2)
        plt.imshow(Image.open(os.path.join(train_set, train_image[sidx])))
    plt.show()










    # os._exit(0)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.90)
    # if torch.cuda.is_available():
    #     print("GPU Ready")
    #     model = model.cuda()
    #
    # for i in range(epoch):
    #     model.train()
    #     for iter, sample in enumerate(tl):
    #         optimizer.zero_grad()
    #         input1 = sample['Images'][0]
    #         feature1 = model(input1)
    #         print(feature1.size)
    #         0
    #         input2 = sample['Images'][1].cuda()
    #         label = sample['Similar'].cuda()
    #         output = model(train=True, input1=input1, input2=input2)
    #         # print(list(output.data.cpu().numpy()))
    #         loss = criterion(output, label.float())
    #         correct = 0
    #         total = 0
    #         print(label, output)
    #         for index, simi in enumerate(output):
    #             if (simi > .5 and label[index] == 1) or (simi < 0.5 and label[index] == 0):
    #                 correct += 1
    #             total += 1
    #         print('epoch:{}=loss:{} accuracy: {}/{}'.format(i, float(loss.data), correct, total))
    #         loss.backward()
    #         optimizer.step()
    #     # scheduler.step()
    #     accuracy = val(tl, model)
    #     print('Epoch: {}    Accuracy: {}'.format(i, accuracy))
    #     torch.save({
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }, '../newcheck/' + str(accuracy) + '.tar')


def val(tl, m):
    m.eval()
    correct = 0
    total = 0
    for iter, sample in enumerate(tl):
        input1 = sample['Images'][0].cuda()
        input2 = sample['Images'][1].cuda()
        label = sample['Similar'].cuda()
        output = m(train=True, input1=input1, input2=input2)
        for index, simi in enumerate(output):
            if (simi > .5 and label[index] == 1) or (simi < 0.5 and label[index] == 0):
                correct += 1
            total += 1
    return float(correct / total * 100)


def inference():
    test_root = '/home/ai/Desktop/project5/dataset/test'
    train_root = '/home/ai/Desktop/project5/dataset/train'
    img_arr = [os.path.join(train_root, img) for img in os.listdir(train_root)]

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    print('load model')
    model = BookModel()
    checkpoint = torch.load('/home/ai/Desktop/project5/newcheck/90.03115264797508.tar')
    model.load_state_dict(checkpoint['model'])
    print('load model done')
    model.eval()
    model = model.cuda()
    ### extrct features
    if not os.path.exists('./features.npy'):
        print('start to extract features, only happen once ')
        arr = []
        for img_name in tqdm(train_image):
            image2_name = os.path.join(train_root, img_name)
            img2 = Image.open(image2_name)
            input2 = torch.unsqueeze(test_transform(img2), dim=0).cuda()
            arr = model(extract_features=True, input2=input2, arr=arr)
        np.save('./features.npy', arr)


    for im1 in os.listdir(train_root):
        plt.figure(figsize=(20,4))
        img1 = Image.open(os.path.join(train_root, im1))
        plt.subplot(1, 6, 1)
        plt.imshow(img1)
        input1 = torch.unsqueeze(test_transform(img1), dim=0).cuda()
        sorted_index = model(input1=input1)

        for iter, index in enumerate(sorted_index):
            plt.subplot(1, 6, iter + 2)
            plt.imshow(Image.open(img_arr[index]))
        plt.show()


def main():
    # train_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((256, 256)),
    #     torchvision.transforms.RandomCrop((224, 224)),
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     torchvision.transforms.ToTensor()
    # ])
    # val_transform = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((224, 224)),
    #     torchvision.transforms.ToTensor()
    # ])
    # print('prepare for datasets')
    # train_ds = BookDataset(train_set, train_transform)
    # val_ds = BookDataset(val_set, val_transform)
    #
    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    # print('datasets done')
    train()
    # inference()

main()
