import json
import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video


class VisDroneVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(VisDroneVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

    # def load_tracker(self, path, tracker_names=None, store=True):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                 if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     for name in tracker_names:
    #         traj_file = os.path.join(path, name, self.name+'.txt')
    #         if self.name == 'KiteSurf':
    #             print(self.name)
    #
    #         if not os.path.exists(traj_file):
    #             if self.name == 'FleetFace':
    #                 txt_name = 'fleetface.txt'
    #             elif self.name == 'Jogging-1':
    #                 txt_name = 'Jogging_1.txt'
    #             elif self.name == 'Jogging-2':
    #                 txt_name = 'Jogging_2.txt'
    #             elif self.name == 'Skating2-1':
    #                 txt_name = 'Skating2_1.txt'
    #             elif self.name == 'Skating2-2':
    #                 txt_name = 'Skating2_2.txt'
    #             elif self.name == 'FaceOcc1':
    #                 txt_name = 'faceocc1.txt'
    #             elif self.name == 'FaceOcc2':
    #                 txt_name = 'faceocc2.txt'
    #             elif self.name == 'Human4-2':
    #                 txt_name = 'Human4_2.txt'
    #             else:
    #                 txt_name = self.name[0].lower()+self.name[1:]+'.txt'
    #             traj_file = os.path.join(path, name, txt_name)
    #         if os.path.exists(traj_file):
    #             with open(traj_file, 'r') as f :
    #                 pred_traj = [list(map(float, x.strip().split('\t')))
    #                         for x in f.readlines()]
    #                 if len(pred_traj) != len(self.gt_traj):
    #                     print(name, len(pred_traj), len(self.gt_traj), self.name)
    #                 if store:
    #                     self.pred_trajs[name] = pred_traj
    #                 else:
    #                     return pred_traj
    #         else:
    #             print(traj_file)
    #     self.tracker_names = list(self.pred_trajs.keys())

class VisDroneDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'OTB100', 'CVPR13', 'OTB50'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(VisDroneDataset, self).__init__(name, dataset_root)

        self.videos = {}
        # using json files
        dataset_root+='2019-part1/VisDrone2018-SOT-train-val'
        json_path = os.path.join(dataset_root, name+'.json')
        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                meta_data = json.load(f)
            # load videos
            videos = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
            for video in videos:
                self.videos[video] = VisDroneVideo(video,
                                                 dataset_root,
                                                 meta_data[video]['video_dir'],
                                                 meta_data[video]['init_rect'],
                                                 meta_data[video]['img_names'],
                                                 meta_data[video]['gt_rect'],
                                                 meta_data[video]['attr'],
                                                 load_img)
        else:
            videos = os.listdir(os.path.join(dataset_root,'sequences'))
            print('VisDrone datasetset, number of sequence:', len(videos))

            for video in videos:
                #print(video)
                init_rect_ = np.loadtxt(os.path.join(dataset_root, 'annotations', video+'.txt'), delimiter=',')
                init_rect_ = list(init_rect_)
                init_rect = init_rect_ #[[None]] * 10000
                init_rect[0] = init_rect_[0]
                gt_rect = init_rect
                _, img_names = self.get_fileNames(os.path.join(dataset_root, 'sequences',video))
                self.videos[video] = VisDroneVideo(video,
                                              dataset_root,
                                              video,
                                              init_rect,
                                              img_names,
                                              gt_rect,
                                              None,
                                              load_img)

        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())


    def get_fileNames(self, rootdir):
        fs = []
        fs_all = []
        for root, dirs, files in os.walk(rootdir, topdown=True):
            files.sort()
            files.sort(key = len)
            if files is not None:
                for name in files:
                    _, ending = os.path.splitext(name)
                    if ending == ".jpg":
                        _, root_ = os.path.split(root)
                        fs.append(os.path.join('sequences' ,root_, name))
                        fs_all.append(os.path.join(root, name))

        return fs_all, fs
