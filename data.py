#*=============================================================================
#
# Author: wukun - 516420282@qq.com
#
# QQ : 516420282
#
# Last modified: 2020-06-08 10:28
#
# Filename: data.py
#
# Description: 
#
#*=============================================================================

from torch.utils.data import Dataset
import os

class YoloDataSet(Dataset):
    def __init__(self, data):
        self.labels = []
        for root, dirs, _ in os.walk(data):
            if len(dirs) > 0:
                for i,d in enumerate(dirs):
                    label = []
                    path = root + d + '/'
                    files = os.listdir(path)
                    for f in files:
                        label.append((path+f, i))
                    self.labels.append(label)

    def __getitem__(self):
        pass

    def __len__(self):
        pass
