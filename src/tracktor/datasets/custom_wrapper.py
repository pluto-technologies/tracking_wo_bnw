import configparser
import csv
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2

from ..config import cfg
from torchvision.transforms import ToTensor


class CustomSequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, seq_name, result_dir, vis_threshold=0.0):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._vis_threshold = vis_threshold
        self.output_dir = result_dir
        self.img_dir = "../../data/" + seq_name
        file_names = sorted(os.listdir(self.img_dir), key = lambda x: x[:4])
        self.file_path = [self.img_dir  + "/" + name for name in file_names]

        self.transforms = ToTensor()

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        img = Image.open(self.file_path[idx]).convert("RGB")
        img = self.transforms(img)
        sample = {}
        sample['img'] = img
        return sample

    def __str__(self):
        return self._seq_name

    def write_results(self, all_tracks):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>, <label>
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1, {}"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(osp.join(self.output_dir, self._seq_name + ".csv"), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    label = int(bb[4])
                    conf = bb[5]
                    writer.writerow(
                        [frame + 1,
                         i + 1,
                         x1 + 1,
                         y1 + 1,
                         x2 - x1 + 1,
                         y2 - y1 + 1,
                         conf, -1, -1, -1, label])

    def load_results(self):
        file_path = osp.join(self.output_dir, self._seq_name)
        results = {}

        if not os.path.isfile(file_path):
            return results

        with open(file_path, "r") as of:
            csv_reader = csv.reader(of, delimiter=',')
            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if not track_id in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = [x1, y1, x2, y2]

        return results

