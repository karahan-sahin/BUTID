import os
import random
import copy
import pickle
import json
import pathlib

import h5py
import torch
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from PIL import Image

# # visualize
from pathlib import Path
import third_party.unisign.utils as utils
from src.config import pose_dirs

# load sub-pose
def load_part_rtm(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]
            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale_2d(np.concatenate([kps, confidences[...,None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale_2d(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[...,2]<=thr] = 0
            
        kps_with_scores[part] = torch.tensor(result)
        
    return kps_with_scores

def load_part_mp(skeletons, force_ok=False):
    kps3d = {}
    scale = None
    
    for part in ['body', 'left', 'right']:
        kps = []
        
        for skeleton in skeletons:
            
            if part == 'body':
                hand_kp2d = skeleton[[15, 16, 17, 18, 19, 20, 21, 58, 59], :]
            elif part == 'left':
                hand_kp2d = skeleton[[20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
            elif part == 'right':
                hand_kp2d = skeleton[[21, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54], :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
            elif part == 'face':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
            
            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            
        kps = np.stack(kps, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale_3d(kps)

            else:
                result, scale, _ = crop_scale_3d(kps)
        else:
            assert not scale is None
            result = kps
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:3] = (result[..., :3]) / scale
                result = np.clip(result, -1, 1)
            
        kps3d[part] = torch.tensor(result)
        
    return kps3d

# input: T, N, 3
# input is un-normed joints
def crop_scale_2d(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]

# input: T, N, 3
# input is un-normed joints
def crop_scale_3d(motion):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion.reshape(-1,3)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    zmin = min(valid_coords[:,2])
    zmax = max(valid_coords[:,2])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin, zmax-zmin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    zs = (zmin+zmax-scale) / 2
    result[...,:3] = (motion[..., :3] - [xs,ys,zs]) / scale
    result[...,:3] = (result[..., :3] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result, scale, [xs,ys,zs]

# build base dataset
class BaseDataset(Dataset.Dataset):
    def collate_fn(self, batch):
        tgt_batch,src_length_batch,name_batch,pose_tmp,gloss_batch,task_batch = [],[],[],[],[],[]
        
        for name_sample, pose_sample, text, gloss, task in batch:
            if pose_sample is None:
                continue
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)
            task_batch.append(task)

        src_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch

            # add task info to src_input
            src_input['tasks'] = task_batch

        tgt_input = {}
        tgt_input['text'] = tgt_batch
        tgt_input['gloss'] = gloss_batch

        tgt_input["labels"] = []
        for text, gloss, task in zip(tgt_batch, gloss_batch, task_batch):
            if task == "ISLR":
                tgt_input["labels"].append(gloss)
            elif task == "SLT":
                tgt_input["labels"].append(text)
            else:
                raise NotImplementedError(f"Unknown task {task}")

        return src_input, tgt_input

class RTMDataset(BaseDataset):
    def __init__(self, path, args, phase):
        super(RTMDataset, self).__init__()
        self.args = args
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase
        
        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[self.args.dataset]
            
        elif "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[self.args.dataset], phase)

        else:
            raise NotImplementedError

        # self.list = list(self.raw_data.keys())
        self.list_key, self.list_task = [], []
        self.tasks = args.tasks if isinstance(args.tasks, list) else [args.task]
        for task in self.tasks:
            self.list_key += list(self.raw_data.keys())
            self.list_task += [task] * len(self.raw_data.keys())

    def __len__(self):
        return len(self.list_key)
    
    def __getitem__(self, index):

        key = self.list_key[index]
        task = self.list_task[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''
        
        name_sample = sample['name']
        pose_sample = self.load_pose(sample['video_path'], sample['start_frame'], sample['end_frame'])

        return name_sample, pose_sample, text, gloss, task
    
    def load_pose(self, path, start, end):
        
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_rtm(skeletons, confs, force_ok=True)
            
        # from pathlib import Path
        # from src.utils.visualization_utils import viz_skeletons
        # os.system('rm ./debug_viz/*.png')
        # viz_skeletons(
        #     kps_with_scores,
        #     Path("./debug_viz/") ,
        #     outfile=f"{path.replace('.pkl', '')}.gif",
        #     pose_type='rtm'
        # )
        # os.system('rm ./debug_viz/*.png')
        # breakpoint()   
        
        return kps_with_scores

    def __str__(self):
        return f'#total {len(self)}'

class BUTIDDataset(BaseDataset):
    def __init__(self, path, args, phase):
        super(BUTIDDataset, self).__init__()
        self.args = args
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase
        
        self.input_type = args.input_type  # 'smplx-j' or 'smplx-p'

        self.pose_dir = pose_dirs[self.args.dataset]
            
        if "BSign22k" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[self.args.dataset], phase)


        # self.list = list(self.raw_data.keys())
        self.list_key, self.list_task = [], []
        self.tasks = args.tasks if isinstance(args.tasks, list) else [args.task]
        for task in self.tasks:
            self.list_key += list(self.raw_data.keys())
            self.list_task += [task] * len(self.raw_data.keys())

    def __len__(self):
        return len(self.list_key)

    def __getitem__(self, index):
        key = self.list_key[index]
        task = self.list_task[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''

        name_sample = sample['name']
        pose_sample = self.load_pose(sample['video_path'], sample['start_frame'], sample['end_frame'])

        return name_sample, pose_sample, text, gloss, task

    def load_pose(self, path, start, end):
        
        pose_path = os.path.join(self.pose_dir, path.replace(".mp4", '.h5'))
        
        if not os.path.exists(pose_path):
            return None
        
        pose = {k: [] for k in ['body', 'left', 'right', 'face']}
        with h5py.File(pose_path, "r") as h5f:
            for frame in range(start, end):
                for part in pose.keys():
                    pose[part].append(h5f[f'{frame:04d}'][part][()])
            
        kps3d = load_part_mp(pose['body'], force_ok=True)
    
        # from pathlib import Path
        # from src.utils.visualization_utils import viz_skeletons
        # os.system('rm ./debug_viz/*.png')
        # viz_skeletons(
        #     kps3d,
        #     Path("./debug_viz/") ,
        #     outfile=f"{path.replace('.h5', '')}_smplx.gif",
        #     pose_type='smplx'
        # )
        # os.system('rm ./debug_viz/*.png')
        # breakpoint()
            
        return kps3d

    def __str__(self):
        return f'#total {len(self)}'

class RTMDatasetForPretraining(BaseDataset):
    def __init__(self, path, args, phase):
        super(RTMDatasetForPretraining, self).__init__()
        self.args = args
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(path)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
       
        if self.args.dataset == "CSL_News":
            self.pose_dir = pose_dirs[self.args.dataset]
      
        else:
            raise NotImplementedError
        
        sum_sample = len(self.annotation)

        if phase == 'train':
            self.start_idx = int(sum_sample * 0.0)
            self.end_idx = int(sum_sample * 0.99)
        else:
            self.start_idx = int(sum_sample * 0.99)
            self.end_idx = int(sum_sample)
        
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, index):
        num_retries = 10  

        # skip some invalid video sample
        for _ in range(num_retries):
            sample = self.annotation[self.start_idx:self.end_idx][index]

            text = sample['text']
            name_sample = sample['video']
           
            try:
                pose_sample = self.load_pose(sample['pose'], sample['video'])
    
            except:
                import traceback

                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            break
           
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return name_sample, pose_sample, text, ''
    
    def load_pose(self, pose_name, rgb_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), 'rb'))
        
        duration = len(pose['scores'])

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp)
            
        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)
        
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
                
        kps_with_scores = load_part_rtm(skeletons, confs)
        
        # from pathlib import Path
        # from src.utils.visualization_utils import viz_skeletons
        # viz_skeletons(
        #     kps_with_scores,
        #     Path("./debug_viz/") ,
        #     outfile=f"{pose_name.replace('.pkl', '')}_rtm.gif",
        #     pose_type='rtm',
        # )
        # os.system('rm ./debug_viz/*.png')
        # breakpoint()        
        
        return kps_with_scores

    def __str__(self):
        return f'#total {len(self)}'
