import os
import sys
import glob
import numpy as np
import time
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import torch
import nibabel as nib
from torch.nn import functional as F
def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_inputs =next(self.loader)
        except StopIteration:
            self.next_inputs = None
            return
        with torch.cuda.stream(self.stream):
            self.next_inputs = [d.cuda(non_blocking=True).float() for d in self.next_inputs]
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_inputs
        self.preload()
        return inputs