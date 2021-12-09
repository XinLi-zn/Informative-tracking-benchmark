import glob
from os.path import join, realpath, dirname
import numpy as np
import time
import torch
import argparse
import json
import os
import cv2
import yaml
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Run your tracker')
    parser.add_argument('--dataset', default='ITB', type=str, help='dataset test')#OTB2015,NFS,UAV123,LASOT, NUS-PRO,VisDrone
    parser.add_argument('--gpunum', default=1, type=int, help='gpu number used')  #
    parser.add_argument('--dataset_path', type=str, help='path to the dataset')  # such as, /home/data/ITB
    args = parser.parse_args()
    return args

def load_dataset(dataset, base_path):
    info = {}
    if 'ITB' in dataset:
        seq_groups = os.listdir(base_path)
        for seq_group in seq_groups:
            if '.' in seq_group:
                continue
            seq_path = os.path.join(base_path,seq_group)
            videos = sorted(os.listdir(seq_path))
            for video in videos:
                video_path = join(seq_path, video)
                if not os.path.isdir(video_path):
                    continue
                image_path = join(video_path, '*.jpg')
                image_files = sorted(glob.glob(image_path))
                gt_path = join(video_path, 'groundtruth.txt')
                gt = np.loadtxt(gt_path, delimiter=',')
                info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'OTB' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        json_path = join(realpath(dirname(__file__)), 'dataset', dataset, dataset + '.json')
        info = json.load(open(json_path, 'r'))

        for v in info.keys():
            path_name = info[v]['video_dir']
            info[v]['image_files'] = [join(base_path, im_f) for im_f in info[v]['img_names']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]
            info[v]['name'] = v
    elif 'UAV123' == dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset, 'data_seq', dataset)
        json_path = join(base_path, dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['video_dir']
            info[v]['image_files'] = [join(base_path, im_f) for im_f in info[v]['img_names']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]
            info[v]['name'] = v
    elif 'NFS' == dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset, 'sequences')
        json_path = join(base_path, dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['video_dir']
            # info[v]['image_files'] = [join(base_path, im_f) for im_f in info[v]['img_names']]
            # for the files without 30 or 120 folders
            info[v]['image_files'] = [join(base_path, path_name, im_f.split('/')[-1]) for im_f in info[v]['img_names']]
            info[v]['gt'] = np.array(info[v]['gt_rect']) - [1, 1, 0, 0]
            info[v]['name'] = v
    elif 'VOTTIR' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.png')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'VOT' in dataset and (not 'VOT2019RGBT' in dataset) and (not 'VOT2020' in dataset):
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'VOT2020' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = open(gt_path, 'r').readlines()
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'RGBT234' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        json_path = join(realpath(dirname(__file__)), 'dataset', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['infrared_imgs'] = [join(base_path, path_name, 'infrared', im_f) for im_f in
                                        info[v]['infrared_imgs']]
            info[v]['visiable_imgs'] = [join(base_path, path_name, 'visible', im_f) for im_f in
                                        info[v]['visiable_imgs']]
            info[v]['infrared_gt'] = np.array(info[v]['infrared_gt'])  # 0-index
            info[v]['visiable_gt'] = np.array(info[v]['visiable_gt'])  # 0-index
            info[v]['name'] = v
    elif 'VOT2019RGBT' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        for video in videos:
            video_path = join(base_path, video)
            in_image_path = join(video_path, 'ir', '*.jpg')
            rgb_image_path = join(video_path, 'color', '*.jpg')
            in_image_files = sorted(glob.glob(in_image_path))
            rgb_image_files = sorted(glob.glob(rgb_image_path))

            assert len(in_image_files) > 0, 'please check RGBT-VOT dataloader'
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            info[video] = {'infrared_imgs': in_image_files, 'visiable_imgs': rgb_image_files, 'gt': gt, 'name': video}
    elif 'VisDrone' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'annotations')
        attr_path = join(base_path, 'attributes')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)
            if not os.path.isdir(video_path):
                continue
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'NUS' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)

        videos = sorted(os.listdir(base_path))
        for video in videos:
            video_path = join(base_path, video)
            if not os.path.isdir(video_path):
                continue
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=' ')
            gt[:, 2:] = gt[:, 2:] - gt[:, :2] + 1
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'VISDRONEVAL' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'annotations')
        attr_path = join(base_path, 'attributes')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)
            if not os.path.isdir(video_path):
                continue
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'VISDRONETEST' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        seq_path = join(base_path, 'sequences')
        anno_path = join(base_path, 'initialization')

        videos = sorted(os.listdir(seq_path))
        for video in videos:
            video_path = join(seq_path, video)

            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(anno_path, '{}.txt'.format(video))
            gt = np.loadtxt(gt_path, delimiter=',').reshape(1, 4)
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'GOT10KVAL' in dataset:
        #         base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        base_path = join('/userhome/TB_project/TracKit/dataset', dataset)
        seq_path = base_path

        #         videos = sorted(os.listdir(seq_path))
        #         videos.remove('list.txt')
        videos = np.loadtxt(join(seq_path, 'list.txt'), dtype=str)
        for video in videos:
            video_path = join(seq_path, video)
            if not os.path.isdir(video_path):
                continue
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'GOT10K' in dataset:  # GOT10K TEST
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset,'test')
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.remove('list.txt')
        for video in videos:
            video_path = join(seq_path, video)
            if not os.path.isdir(video_path):
                continue
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': [gt], 'name': video}
    elif 'TrackingNet' in dataset:  # GOT10K TEST
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset,'TEST/frames')
        seq_path = base_path

        videos = sorted(os.listdir(seq_path))
        videos.reverse()
        #videos.remove('list.txt')
        for video in videos:
            video_path = join(seq_path, video)
            if not os.path.isdir(video_path):
                continue
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, '../../anno',video+'.txt')
            gt = np.loadtxt(gt_path, delimiter=',')
            info[video] = {'image_files': image_files, 'gt': [gt], 'name': video}
    elif 'LASOT' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', dataset)
        # json_path = join(realpath(dirname(__file__)), 'dataset', dataset + '.json')
        # jsons = json.load(open(json_path, 'r'))
        # testingvideos = list(jsons.keys())

        videos = sorted(os.listdir(base_path))
        for video in videos:
            print(video)
            video_path = join(base_path, video)
            if not os.path.isdir(video_path):
                continue
            # ground truth
            gt_path = join(video_path, 'groundtruth.txt')
            if not os.path.exists(gt_path):
                continue
            gt = np.loadtxt(gt_path, delimiter=',')
            gt = gt - [1, 1, 0, 0]
            # get img file
            img_path = join(video_path, 'img', '*jpg')
            image_files = sorted(glob.glob(img_path))

            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'DAVIS' in dataset and 'TEST' not in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', 'DAVIS')
        list_path = join(realpath(dirname(__file__)), 'dataset', 'DAVIS', 'ImageSets', dataset[-4:],
                         'val.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            info[video] = {}
            info[video]['anno_files'] = sorted(glob.glob(join(base_path, 'Annotations/480p', video, '*.png')))
            info[video]['image_files'] = sorted(glob.glob(join(base_path, 'JPEGImages/480p', video, '*.jpg')))
            info[video]['name'] = video
    elif 'YTBVOS' in dataset:
        base_path = join(realpath(dirname(__file__)), 'dataset', 'YTBVOS', 'valid')
        json_path = join(realpath(dirname(__file__)), 'dataset', 'YTBVOS', 'valid', 'meta.json')
        meta = json.load(open(json_path, 'r'))
        meta = meta['videos']
        info = dict()
        for v in meta.keys():
            objects = meta[v]['objects']
            frames = []
            anno_frames = []
            info[v] = dict()
            for obj in objects:
                frames += objects[obj]['frames']
                anno_frames += [objects[obj]['frames'][0]]
            frames = sorted(np.unique(frames))
            info[v]['anno_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in frames]
            info[v]['anno_init_files'] = [join(base_path, 'Annotations', v, im_f + '.png') for im_f in anno_frames]
            info[v]['image_files'] = [join(base_path, 'JPEGImages', v, im_f + '.jpg') for im_f in frames]
            info[v]['name'] = v

            info[v]['start_frame'] = dict()
            info[v]['end_frame'] = dict()
            for obj in objects:
                start_file = objects[obj]['frames'][0]
                end_file = objects[obj]['frames'][-1]
                info[v]['start_frame'][obj] = frames.index(start_file)
                info[v]['end_frame'][obj] = frames.index(end_file)
    else:
        raise ValueError("Dataset not support now, edit for other dataset youself...")
    return info

def track(dataset,args):
    toc=0
    # initialization of your tracker
    # model = build_vgg16(cfg)
    # tracker = Tadt_Tracker(cfg, model=model, device='cuda', display=False)
    assert False, 'you may initialize your model here'

    #setting the result file path
    result_path = os.path.join('result', args.dataset,'your-method')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    video_keys = list(dataset.keys()).copy()
    video_keys.reverse()
    for video_key in video_keys:
        print(video_key,'------------')
        video=dataset[video_key]
        result_file = os.path.join(result_path, '{:s}.txt'.format(video['name']))
        if os.path.exists(result_file):
            print(result_file + ' is existed!')
            continue

        image_files, gt = video['image_files'], video['gt']
        regions = []
        lost = 0
        tic = cv2.getTickCount()
        for i,image_file in enumerate(image_files):
            if 0==i:
                # process in the initial frame
                # tracker.initialize_tadt(image_files[0], gt[0])
                assert False, 'process the initial frame here'
                regions.append(gt[0])
                scale_flag=1
            else:
                # process in each following frame
                #location, scale_flag = tracker.tracking(image_file, i, scale_flag)
                assert False, 'process each following frame here'
                regions.append(location)


        toc += cv2.getTickCount() - tic
        with open(result_file, "w") as fin:
            if 'VOT' in args.dataset or 'vot' in args.dataset:
                for x in regions:
                    if isinstance(x, int):
                        fin.write("{:d}\n".format(x))
                    else:
                        p_bbox = x.copy()
                        fin.write(','.join([str(i) for i in p_bbox]) + '\n')
            elif 'ITB' in args.dataset or'TrackingNet' in args.dataset or'OTB' in args.dataset or 'LASOT' in args.dataset or 'UAV' in args.dataset or 'NFS' in args.dataset or 'lasot' in args.dataset or 'NUS' in args.dataset:
                for x in regions:
                    p_bbox = x.copy()
                    fin.write(
                        ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
            elif 'VISDRONE' in args.dataset or 'GOT10K' in args.dataset or 'VisDrone' in args.dataset:
                for x in regions:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for idx, i in enumerate(p_bbox)]) + '\n')
            else:
                assert False, 'dataset for testing is unknown, in test_tadt.py line 359'
        toc /= cv2.getTickFrequency()
        print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video['name'], toc, i / toc, lost))


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.dataset,args.dataset_path)
    track(dataset, args)

# you may run this file using the following command
# CUDA_VISIBLE_DEVICES=0 python test_tracker.py --dataset ITB --dataset_path /path-to/ITB


