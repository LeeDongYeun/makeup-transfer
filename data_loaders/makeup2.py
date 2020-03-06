import os
import torch
import random
import linecache
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class MAKEUP2(Dataset):
    def __init__(self, image_path, transform, mode, transform_mask, cls_list):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.transform_mask = transform_mask

        self.cls_list = cls_list
        self.cls_A = cls_list[0]
        self.cls_B = cls_list[1]

        for cls in self.cls_list:
            setattr(self, "train_" + cls + "_list_path", os.path.join(self.image_path, "train_" + cls + ".txt"))
            setattr(self, "train_" + cls + "_lines", open(getattr(self, "train_" + cls + "_list_path"), 'r').readlines())
            setattr(self, "num_of_train_" + cls + "_data", len(getattr(self, "train_" + cls + "_lines")))

        for cls in self.cls_list:
            if self.mode == "test_all":
                setattr(self, "test_" + cls + "_list_path", os.path.join(self.image_path, "test_" + cls + "_all.txt"))
                setattr(self, "test_" + cls + "_lines", open(getattr(self, "test_" + cls + "_list_path"), 'r').readlines())
                setattr(self, "num_of_test_" + cls + "_data", len(getattr(self, "test_" + cls + "_lines")))
            else:
                setattr(self, "test_" + cls + "_list_path", os.path.join(self.image_path, "test_" + cls + ".txt"))
                setattr(self, "test_" + cls + "_lines", open(getattr(self, "test_" + cls + "_list_path"), 'r').readlines())
                setattr(self, "num_of_test_" + cls + "_data", len(getattr(self, "test_" + cls + "_lines")))

        print('Makeup dataloader 2')
        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')
    
    def preprocess(self):
        for cls in self.cls_list:
            setattr(self, "train_" + cls + "_filenames", [])
            setattr(self, "train_" + cls + "_mask_filenames", [])
            setattr(self, "train_" + cls + "_ids", [])
            setattr(self, "train_" + cls + "_classes", [])

            lines = getattr(self, "train_" + cls + "_lines")
            random.shuffle(lines)

            for i, line in enumerate(lines):
                splits = line.split()
                getattr(self, "train_" + cls + "_filenames").append(splits[0])
                getattr(self, "train_" + cls + "_mask_filenames").append(splits[1])
                getattr(self,  "train_" + cls + "_ids").append(splits[2])
                getattr(self,  "train_" + cls + "_classes").append(splits[3])
            
            print(getattr(self,  "train_" + cls + "_ids"))
        
        for cls in self.cls_list:
            setattr(self, "test_" + cls + "_filenames", [])
            setattr(self, "test_" + cls + "_mask_filenames", [])
            setattr(self, "test_" + cls + "_ids", [])
            setattr(self, "test_" + cls + "_classes", [])

            lines = getattr(self, "test_" + cls + "_lines")
            for i, line in enumerate(lines):
                splits = line.split()
                getattr(self, "test_" + cls + "_filenames").append(splits[0])
                getattr(self, "test_" + cls + "_mask_filenames").append(splits[1])
                getattr(self,  "test_" + cls + "_ids").append(splits[2])
                getattr(self,  "test_" + cls + "_classes").append(splits[3])

        if self.mode == "test_baseline":
            setattr(self, "test_" + self.cls_A + "_filenames", os.listdir(os.path.join(self.image_path, "baseline", "org_aligned")))
            setattr(self, "num_of_test_" + self.cls_A + "_data", len(os.listdir(os.path.join(self.image_path, "baseline", "org_aligned"))))
            setattr(self, "test_" + self.cls_B + "_filenames", os.listdir(os.path.join(self.image_path, "baseline", "ref_aligned")))
            setattr(self, "num_of_test_" + self.cls_B + "_data", len(os.listdir(os.path.join(self.image_path, "baseline", "ref_aligned"))))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self.mode == 'train' or self.mode == 'train_finetune':
            # Assign image A
            image_A = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_A + "_filenames")[index])).convert("RGB")
            mask_A = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_A + "_mask_filenames")[index]))
            id_A = getattr(self, "train_" + self.cls_A + "_ids")[index]
            class_A = getattr(self, "train_" + self.cls_A + "_classes")[index]

            # Assign image B
            id_B = -1
            class_B = class_A
            while (id_A != id_B and class_A == class_B):
                index_B = random.randint(0, getattr(self, "num_of_train_" + self.cls_B + "_data") - 1)
                id_B = getattr(self, "train_" + self.cls_B + "_ids")[index_B]
                class_B = getattr(self, "train_" + self.cls_B + "_classes")[index_B]
            image_B = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_B + "_filenames")[index_B])).convert("RGB")
            mask_B = Image.open(os.path.join(self.image_path, getattr(self, "train_" + self.cls_B + "_mask_filenames")[index_B]))
        
            if self.transform:
                image_A = self.transform(image_A)
                mask_A = self.transform_mask(mask_A)
                image_B = self.transform(image_B)
                mask_B = self.transform_mask(mask_B)
            
            sample = {'image_A' : image_A, 'mask_A':mask_A, 'id_A' : id_A, 'class_A' : class_A,
                        'image_B' : image_B, 'mask_B':mask_B, 'id_B' : id_B, 'class_B' : class_B}
            return sample
        
        '''
        if self.mode in ['test', 'test_all']:
            #"""
            image_A = Image.open(os.path.join(self.image_path, getattr(self, "test_" + self.cls_A + "_filenames")[index // getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')])).convert("RGB")
            image_B = Image.open(os.path.join(self.image_path, getattr(self, "test_" + self.cls_B + "_filenames")[index % getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')])).convert("RGB")
            return self.transform(image_A), self.transform(image_B)
        if self.mode == "test_baseline":
            image_A = Image.open(os.path.join(self.image_path, "baseline", "org_aligned", getattr(self, "test_" + self.cls_A + "_filenames")[index // getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')])).convert("RGB")
            image_B = Image.open(os.path.join(self.image_path, "baseline", "ref_aligned", getattr(self, "test_" + self.cls_B + "_filenames")[index % getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')])).convert("RGB")
            return self.transform(image_A), self.transform(image_B)
        '''

    def __len__(self):
        if self.mode == 'train' or self.mode == 'train_finetune':
            num_A = getattr(self, 'num_of_train_' + self.cls_list[0] + '_data')
            num_B = getattr(self, 'num_of_train_' + self.cls_list[1] + '_data')
            return num_A #max(num_A, num_B)
        '''
        elif self.mode in ['test', "test_baseline", 'test_all']:
            num_A = getattr(self, 'num_of_test_' + self.cls_list[0] + '_data')
            num_B = getattr(self, 'num_of_test_' + self.cls_list[1] + '_data')
            return num_A * num_B
        ''' 