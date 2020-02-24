import os
import torch
import random
import linecache

from torch.utils.data import Dataset
from PIL import Image

class MAKEUP2(Dataset):
    def __init__(self, image_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        # self.transform_mask = transform_mask

        self.train_list_path = os.path.join(self.image_path, "train" + ".txt")
        self.train_lines = open(getattr(self, "train" + "_list_path"), 'r').readlines()
        self.num_of_train_data = len(getattr(self, "train" + "_lines"))

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')
    
    def preprocess(self):
        self.train_filenames = []
        self.train_ids = []
        self.train_classes = []
        
        lines = self.train_lines
        random.suffle(lines)

        for i line in enumerate(lines):
            splits = line.split()
            self.train_filenames.append(splits[0])
            self.train_ids.append(splits[1])
            self.train_classes.append(splits[2])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_A = Image.open(os.path.join(self.image_path, self.train_filenames[index])).convert("RGB")
        id_A = self.train_ids[index]
        class_A = self.train_classes[index]

        # Assign image B
        id_B = -1
        class_B = class_A
        while (id_A != id_B and class_A == class_B):
            index_B = random.randint(0, self.num_of_train_data - 1)
            id_B = self.train_ids[index_B]
            class_B = self.train_classes[index_B]
        image_B = Image.open(os.path.join(self.image_path, self.train_filenames[index_B])).convert("RGB")
        
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
        
        sample = {'image_A' : image_A, 'id_A' : id_A, 'class_A' : class_A,
                    'image_B' : image_B, 'id_B' : id_B, 'class_B' : class_B}

        return sample
    
    def __len__(self):
        return self.num_of_train_data
        
            