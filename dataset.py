import os
import torch.nn as nn
from torch.utils.data import Dataset
import random
from PIL import Image


class BookDataset(Dataset):
    def __init__(self, path, transform = None):
        super(BookDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.dict = self.get_dict()
        # print(self.dict)
    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self, item):
        files = os.listdir(self.path)
        file1 = os.path.join(self.path,files[item])
        key = '_'.join(file1.split('_')[1:3])
        choice = random.randint(0,1)
        if choice == 1:
            while True:
                file2 = self.dict[key][random.randint(0, len(self.dict[key])-1)]
                file2 = os.path.join(self.path,file2)
                if file1!=file2:
                    break
        else:
            while True:
                file2 = os.path.join(self.path,files[random.randint(0,self.__len__()-1)])
                if file1 != file2:
                    break
        # print(file1, file2, choice)
        # os._exit(0)
        img1 = self.transform(Image.open(file1))
        img2 = self.transform(Image.open(file2))
        imgs = [img1,img2]
        return {"Images": imgs, "Similar": choice}

    def get_dict(self):
        files = os.listdir(self.path)
        dict = {}
        for i in files:
            key = '_'.join(i.split('_')[1:3])
            # print(key)
            if dict.get(key,'')=='':
                array = []
                array.append(i)
                dict[key] = array
            else:
                dict[key].append(i)
        return dict
if __name__ == '__main__':
    bd = BookDataset('../dataset/train/', )
    print(bd.__len__())
