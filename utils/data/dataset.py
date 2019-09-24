import pandas as pd
import numpy as np
import os
from scipy import ndimage
from torch.utils.data import Dataset
import torch

from utils.bbox import *

class NoduleDataset(Dataset):
    def __init__(self, ct_dir, bbox_csv_path, label_csv_path, **kwargs):
        """
        Returns:
            nodules: list of Tensor
            label: int
            [bboxes: list of Tensor]
            [scan: str]
        """
        
        self.ct_dir = ct_dir
        self.bbox_csv = pd.read_csv(bbox_csv_path, dtype={'id': object})
        self.label_csv = pd.read_csv(label_csv_path, dtype={'id': object})
        
        expected_kw = {'iou_th', 'pad', 'ret_bbox', 'ret_id', 'out_dia', 'ignore_bbox_d', 'skip_missed_npy'}
        
        if not set(kwargs.keys()) <= expected_kw:
            raise TypeError('unexpected keywords: ' + str(set(kwargs.keys()) - expected_kw))
        
        self.iou_th = kwargs.get('iou_th', 0.05)
        self.pad = kwargs.get('pad', 170)
        self.ret_bbox = kwargs.get('ret_bbox', False)
        self.ret_id = kwargs.get('ret_id', False)
        self.out_dia = kwargs.get('out_dia', 64)
        self.ignore_bbox_d = kwargs.get('ignore_bbox_d', False)
        self.skip_missed_npy = kwargs.get('skip_missed_npy', False)
        
        self.ids = []
        self.nmsed_bboxes = {}
        self.nodules = {}
        
        for idx in list(self.label_csv.id):
            #c z x y            
            path = os.path.join(self.ct_dir, "{}_clean.npy").format(idx)
            
            if not os.path.exists(path):
                if not self.skip_missed_npy:
                    raise FileNotFoundError("[Errno 2] No such file or directory: '{}'".format(path))
                else:
                    print("skip "+ path)
                    continue

            self.ids.append(idx)
        
    def __getitem__(self, idx):
        #idx: int
        #scan_id: str
        
        scan_id = self.ids[idx]
        
        if scan_id not in self.nmsed_bboxes:
            self._processOneScan(scan_id)
        
        ret_data = (self.nodules[scan_id], int(self.label_csv.label[idx]))
        
        if self.ret_bbox:
            ret_data = (*ret_data, self.nmsed_bboxes[scan_id])
            
        if self.ret_id:
            ret_data = (*ret_data, scan_id)
            
        return ret_data
    
    def __len__(self):
        return len(self.ids)
        
    def crop(self, scan, bbox, pad=None, out_dia=64, ignore_bbox_d=False):
        # bbox.shape (4,)
        # z x y d
        # scan.shape (Num_slice, Height, Width)
        
        assert bbox.shape == (4, ), "Wrong bbox.shape" + str(bbox.shape)
        assert len(scan.shape) == 3, "Wrong scan dimension"

        r = out_dia // 2 if ignore_bbox_d else int(bbox[3] / 2)
        
        bbox = bbox.astype('int')

        patch = scan[np.maximum(0, bbox[0] - r):np.minimum(scan.shape[0], bbox[0] + r),
                     np.maximum(0, bbox[1] - r):np.minimum(scan.shape[1], bbox[1] + r),
                     np.maximum(0, bbox[2] - r):np.minimum(scan.shape[2], bbox[2] + r)]

        padz = (np.maximum(0, r - bbox[0]), np.maximum(0, bbox[0] + r - scan.shape[0]))
        padx = (np.maximum(0, r - bbox[1]), np.maximum(0, bbox[1] + r - scan.shape[1]))
        pady = (np.maximum(0, r - bbox[2]), np.maximum(0, bbox[2] + r - scan.shape[2]))

        if pad is None:
            return patch

        patch = np.pad(patch, (padz, padx, pady), 'constant', constant_values=pad)

        patch = ndimage.zoom(patch, (out_dia / patch.shape[0], out_dia / patch.shape[1], out_dia / patch.shape[2]))
        
        return patch
    
    def _processOneScan(self, idx):
        scan = np.load(os.path.join(self.ct_dir, "{}_clean.npy").format(idx))[0]
        
        bboxes = self.bbox_csv[self.bbox_csv.id == idx].iloc[:, 1:].to_numpy()
        nmsed_bboxes = nms(bboxes, self.iou_th)

        nodules_one_scan = [torch.Tensor(self.crop(scan, bbox[1:], self.pad, self.out_dia, self.ignore_bbox_d)) for bbox in nmsed_bboxes]

        self.nmsed_bboxes[idx] = torch.Tensor(nmsed_bboxes)
        self.nodules[idx] = nodules_one_scan
