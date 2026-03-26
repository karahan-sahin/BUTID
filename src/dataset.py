import os
import random
import copy
import pickle
import json
import pathlib
import pandas as pd

import h5py
import torch
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from PIL import Image

# # visualize
from pathlib import Path
from src.config import pose_dirs, vq_dirs

import sys

sys.path.append("./third_party/unisign")
import third_party.unisign.utils as utils

import signal

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()

def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
    return result

BODY_IDXS_MP = [0, 7, 8, 11, 12, 13, 14, 15, 16]
HAND_IDXS_MP = list(range(21))
FACE_IDXS_MP = [ 162, 93, 172, 149, 152, 378, 397, 323, 389, 78, 82, 13, 312, 308, 317, 14, 87, 4 ]

# load sub-pose
def load_part_rtm(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None

    for part in ["body", "left", "right", "face"]:
        kps = []
        confidences = []

        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]

            if part == "body":
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == "left":
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == "right":
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == "face":
                hand_kp2d = skeleton[
                    [i for i in list(range(23, 23 + 17))[::2]]
                    + [i for i in range(83, 83 + 8)]
                    + [53],
                    :,
                ]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[
                    [i for i in list(range(23, 23 + 17))[::2]]
                    + [i for i in range(83, 83 + 8)]
                    + [53]
                ]
            else:
                raise NotImplementedError

            kps.append(hand_kp2d)
            confidences.append(confidence)

        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)

        if part == "body":
            if force_ok:
                result, scale, _ = crop_scale_2d(
                    np.concatenate([kps, confidences[..., None]], axis=-1), thr
                )

            else:
                result, scale, _ = crop_scale_2d(
                    np.concatenate([kps, confidences[..., None]], axis=-1), thr
                )
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[..., 2] <= thr] = 0

        kps_with_scores[part] = torch.tensor(result)

    return kps_with_scores


def load_part_mp(skeletons, force_ok=False):
    kps3d = {}
    scale = None
    null_default = {
        "body": np.zeros((33, 3)),
        "left": np.zeros((21, 3)),
        "right": np.zeros((21, 3)),
        "face": np.zeros((468, 3)),
    }

    for part in ["body", "left", "right", "face"]:

        kps = []
        for skeleton in skeletons:

            skeleton = (
                skeleton[part]
                if part in skeleton.keys() and skeleton[part].shape[0] != 0
                else null_default[part]
            )

            if part == "body":
                hand_kp2d = skeleton[BODY_IDXS_MP, :]

            elif part == "left":
                hand_kp2d = skeleton[HAND_IDXS_MP, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]  # wrist normalize

            elif part == "right":
                hand_kp2d = skeleton[HAND_IDXS_MP, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]  # wrist normalize

            elif part == "face":
                hand_kp2d = skeleton[FACE_IDXS_MP, :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]  # nose tip normalize

            else:
                raise NotImplementedError

            kps.append(hand_kp2d)

        kps = np.stack(kps, axis=0)

        if part == "body":
            if force_ok:
                result, scale, _ = crop_scale_3d(kps)

            else:
                result, scale, _ = crop_scale_3d(kps)
        else:
            assert not scale is None
            result = kps
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :3] = (result[..., :3]) / scale
                result = np.clip(result, -1, 1)

        kps3d[part] = torch.tensor(result)

    return kps3d


# input: T, N, 3
# input is un-normed joints
def crop_scale_2d(motion, thr):
    """
    Motion: [(M), T, 17, 3].
    Normalize to [-1, 1]
    """
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]


# input: T, N, 3
# input is un-normed joints
def crop_scale_3d(motion):
    """
    Motion: [(M), T, 17, 3].
    Normalize to [-1, 1]
    """
    result = copy.deepcopy(motion)
    valid_coords = motion.reshape(-1, 3)
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    zmin = min(valid_coords[:, 2])
    zmax = max(valid_coords[:, 2])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax - xmin, ymax - ymin, zmax - zmin) * ratio
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    zs = (zmin + zmax - scale) / 2
    result[..., :3] = (motion[..., :3] - [xs, ys, zs]) / scale
    result[..., :3] = (result[..., :3] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result, scale, [xs, ys, zs]


# build base dataset
class BaseDataset(Dataset.Dataset):
    def collate_fn(self, batch):
        tgt_batch, src_length_batch, name_batch, pose_tmp, vq_tmp, gloss_batch, task_batch = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for name_sample, pose_sample, vq_sample, text, gloss, task in batch:

            if pose_sample is None and self.args.input_mode == "pose":
                continue
            if vq_sample is None and self.args.input_mode == "vq":
                continue

            name_batch.append(name_sample)
            pose_tmp.append(load_part_mp(pose_sample) if self.args.input_mode == "pose" else None)
            vq_tmp.append(vq_sample if self.args.input_mode == "vq" else None)
            tgt_batch.append(text)
            gloss_batch.append(gloss)
            task_batch.append(task)

        src_input = {}

        if self.args.input_mode == "pose":
            keys = pose_tmp[0].keys()
            for key in keys:

                max_len = max([len(vid[key]) for vid in pose_tmp])
                video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])

                padded_video = [
                    torch.cat(
                        (
                            vid[key],
                            vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                        ),
                        dim=0,
                    )
                    for vid in pose_tmp
                ]

                img_batch = torch.stack(padded_video, 0)

                src_input[key] = img_batch
                if "attention_mask" not in src_input.keys():
                    src_length_batch = video_length

                    mask_gen = []
                    for i in src_length_batch:
                        tmp = torch.ones([i]) + 7
                        mask_gen.append(tmp)
                    mask_gen = pad_sequence(mask_gen, padding_value=0, batch_first=True)
                    img_padding_mask = (mask_gen != 0).long()
                    src_input["attention_mask"] = img_padding_mask

                    src_input["name_batch"] = name_batch
                    src_input["src_length_batch"] = src_length_batch

        elif self.args.input_mode == "vq":
            # vq_tmp: list of numpy arrays with shape [4, T]
            vq_tensors = [
                torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for v in vq_tmp
            ]
            # vq_tensors: list of [4, T_i]
            src_length_batch = torch.LongTensor([v.shape[1] for v in vq_tensors])
            max_len = src_length_batch.max().item()

            # Pad each [4, T_i] -> [4, max_len] with 0, then stack to [B, 4, max_len]
            vq_padded = torch.stack(
                [
                    torch.nn.functional.pad(v, (0, max_len - v.shape[1]), value=0)
                    for v in vq_tensors
                ],
                dim=0,
            )

            mask_gen = pad_sequence(
                [torch.ones(l) for l in src_length_batch],
                batch_first=True,
                padding_value=0,
            )
            # partition vq into body, left, right, face — [B, T] for nn.Embedding
            src_input["body"]  = vq_padded[:, 0, :].long()  # [B, T]
            src_input["left"]  = vq_padded[:, 1, :].long()  # [B, T]
            src_input["right"] = vq_padded[:, 2, :].long()  # [B, T]
            src_input["face"]  = vq_padded[:, 3, :].long()  # [B, T]

            src_input["attention_mask"] = mask_gen.long()  # [B, T_max]
            src_input["name_batch"] = name_batch
            src_input["src_length_batch"] = src_length_batch

        # add task info to src_input
        src_input["tasks"] = task_batch

        tgt_input = {}
        tgt_input["text"] = tgt_batch
        tgt_input["gloss"] = gloss_batch

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

        text = sample["text"]
        if "gloss" in sample.keys():
            gloss = " ".join(sample["gloss"])
        else:
            gloss = ""

        name_sample = sample["name"]
        pose_sample = self.load_pose(
            sample["video_path"], sample["start_frame"], sample["end_frame"]
        )

        return name_sample, pose_sample, text, gloss, task

    def load_pose(self, path, start, end):

        pose = pickle.load(
            open(os.path.join(self.pose_dir, path.replace(".mp4", ".pkl")), "rb")
        )

        if "start" in pose.keys():
            assert pose["start"] < pose["end"]
            duration = pose["end"] - pose["start"]
            start = pose["start"]
        else:
            duration = len(pose["scores"])
            start = 0

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))

        tmp = np.array(tmp) + start

        skeletons = pose["keypoints"]
        confs = pose["scores"]
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
        return f"#total {len(self)}"


class BUTIDDataset(BaseDataset):
    def __init__(self, path, args, phase):
        super(BUTIDDataset, self).__init__()
        self.args = args
        self.phase = phase
        self.max_length = args.max_length

        self.raw_data = pd.read_csv(path)
        # self.raw_data = self.raw_data.sample(frac=1).reset_index(drop=True)
        self.raw_data = self.raw_data.to_dict(orient="records")

        self.pose_dir = pose_dirs[self.args.dataset]
        self.vq_dir = vq_dirs[self.args.dataset]

        if "BSign22k" in self.args.dataset or "AUTSL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[self.args.dataset], phase)
            self.vq_dir = os.path.join(vq_dirs[self.args.dataset], phase)
        
        # check existing vq files and filter raw_data accordingly
        existing_vq_files = set(os.listdir(self.vq_dir))
        filtered_raw_data = []
        for sample in self.raw_data:
            video_id = sample["video_id"]
            vq_filename = f"{video_id}.pkl"
            if vq_filename in existing_vq_files:
                filtered_raw_data.append(sample)
            else:
                pass
                # print(f"Warning: VQ file {vq_filename} not found for video_id {video_id}. Skipping this sample.")

        self.raw_data = filtered_raw_data

        self.tasks = args.tasks if isinstance(args.tasks, list) else [args.task]
        self.list_key, self.list_task = [], []
        for task in self.tasks:
            self.list_key += list(range(len(self.raw_data)))
            self.list_task += [task] * len(self.raw_data)

        self.start_pad = getattr(args, "start_pad", 0)
        self.end_pad = getattr(args, "end_pad", 0)

        # Store only paths, not open file handles
        self.h5_paths = {
            sample["video_id"]: os.path.join(self.pose_dir, f"{sample['video_id']}.h5")
            for sample in self.raw_data
        }

        # Small lazy-open cache per worker/process
        self._h5_cache = {}
        self._h5_cache_order = []
        self._max_open_h5 = getattr(args, "max_open_h5", 8)

    def _get_h5(self, video_id):
        path = self.h5_paths.get(video_id)
        if path is None or not os.path.exists(path):
            return None

        if video_id in self._h5_cache:
            return self._h5_cache[video_id]

        f = h5py.File(path, "r")
        self._h5_cache[video_id] = f
        self._h5_cache_order.append(video_id)

        # small FIFO/LRU-ish eviction
        if len(self._h5_cache_order) > self._max_open_h5:
            old_vid = self._h5_cache_order.pop(0)
            old_f = self._h5_cache.pop(old_vid, None)
            if old_f is not None:
                try:
                    old_f.close()
                except Exception:
                    pass

        return f

    def _close_h5_cache(self):
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache = {}
        self._h5_cache_order = []

    def __del__(self):
        self._close_h5_cache()

    def __len__(self):
        return len(self.list_key)

    def __getitem__(self, index):
        raw_idx = self.list_key[index]
        task = self.list_task[index]
        sample = self.raw_data[raw_idx]

        key = sample["caption_id"]
        video_id = sample["video_id"]
        text = sample["text"]

        if "gloss" in sample and pd.notna(sample["gloss"]):
            gloss_val = sample["gloss"]
            if isinstance(gloss_val, list):
                gloss = " ".join(gloss_val)
            else:
                gloss = str(gloss_val)
        else:
            gloss = ""

        TIMEOUT = getattr(self.args, "io_timeout", 5)  # seconds

        try:
            if self.args.online_data_loading:
                pose_sample = (
                    run_with_timeout(
                        self.load_pose, TIMEOUT, video_id, sample["start"], sample["end"]
                    )
                    if self.args.input_mode == "pose"
                    else None
                )

                vq_sample = (
                    run_with_timeout(self.load_vq, TIMEOUT, key)
                    if self.args.input_mode == "vq"
                    else None
                )

            else:
                pose_path = os.path.join(self.pose_dir, video_id, f"{key}.pt")

                loaded = run_with_timeout(torch.load, TIMEOUT, pose_path, map_location="cpu")
                pose_sample = loaded["pose"]

                vq_sample = run_with_timeout(
                    self.load_vq, TIMEOUT, video_id, sample["start"], sample["end"]
                )

        except TimeoutException:
            print(f"[TIMEOUT] Skipping sample {key}")
            return key, None, None, text, gloss, task

        except Exception as e:
            print(f"[ERROR] {key}: {e}")
            return key, None, None, text, gloss, task

        # Normalize/convert here so collate_fn does less CPU work
        # Only convert if it's still in raw mediapipe-list form
        # if isinstance(pose_sample, list):
        #     pose_sample = load_part_mp(pose_sample)

        if self.args.debug:
            from pathlib import Path
            from src.utils.visualization_utils import viz_skeletons

            os.system("rm -f ./debug_viz/*.png")
            viz_skeletons(
                pose_sample,
                Path("./debug_viz/"),
                outfile=f"{key}_mp.gif",
                title=f"{key} - Translation: {text}",
                pose_type="mp",
            )
            os.system("rm -f ./debug_viz/*.png")
            breakpoint()

        return key, pose_sample, vq_sample, text, gloss, task

    def load_pose(self, video_id, start, end):
        h5f = self._get_h5(video_id)
        if h5f is None:
            print(f"Pose file for {video_id} does not exist.")
            return None

        if "keypoints" not in h5f:
            print(f"'keypoints' group missing in {video_id}.")
            return None

        pose = []
        start = max(0, int(start) - int(self.start_pad * 25))
        end = int(end) + int(self.end_pad * 25)

        keypoints_grp = h5f["keypoints"]

        for frame in range(start, end):
            frame_grp = keypoints_grp.get(f"frame_{frame:04d}")
            if frame_grp is None:
                continue

            pose.append(
                {
                    "body": frame_grp["pose_landmarks"][:] if "pose_landmarks" in frame_grp else np.zeros((33, 3), dtype=np.float32),
                    "left": frame_grp["left_hand_landmarks"][:] if "lefts_hand_landmarks" in frame_grp else np.zeros((21, 3), dtype=np.float32),
                    "right": frame_grp["right_hand_landmarks"][:] if "right_hand_landmarks" in frame_grp else np.zeros((21, 3), dtype=np.float32),
                    "face": frame_grp["face_landmarks"][:] if "face_landmarks" in frame_grp else np.zeros((468, 3), dtype=np.float32),
                }
            )

        if len(pose) == 0:
            return None

        # Keep temporal order
        if len(pose) > self.max_length:
            idx = sorted(random.sample(range(len(pose)), self.max_length))
            pose = [pose[i] for i in idx]

        return pose

    def load_vq(self, key):
        video_id = key.split(".")[0] # video_id 
        index = int(key.split(".")[-1]) - 1  # Assuming caption_id is like "video_id.0001", "video_id.0002", etc.
        vq_path = os.path.join(self.vq_dir, f"{video_id}.pkl")
        if not os.path.exists(vq_path):
            print(f"VQ file for {vq_path} does not exist.")
            return None
        try:
            with open(vq_path, "rb") as f:
                vq_data = pickle.load(f)

            if index < len(vq_data):
                data = vq_data[index]
                # filter max length
                if data.shape[1] > self.max_length:
                    idx = sorted(random.sample(range(data.shape[1]), self.max_length))
                    data = data[:, idx]
                return data
            else:
                print(f"Index {index} out of range for VQ data in {video_id}.")
                return None
        except Exception as e:
            print(f"Error loading VQ data for {video_id}: {e}")

        return None

    def __str__(self):
        return f"#total {len(self)}"

class RTMDatasetForPretraining(BaseDataset):
    def __init__(self, path, args, phase):
        super(RTMDatasetForPretraining, self).__init__()
        self.args = args
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(path)

        with path.open(encoding="utf-8") as f:
            self.annotation = json.load(f)

        if self.args.dataset == "CSL_News":
            self.pose_dir = pose_dirs[self.args.dataset]

        else:
            raise NotImplementedError

        sum_sample = len(self.annotation)

        if phase == "train":
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
            sample = self.annotation[self.start_idx : self.end_idx][index]

            text = sample["text"]
            name_sample = sample["video"]

            try:
                pose_sample = self.load_pose(sample["pose"], sample["video"])

            except:
                import traceback

                traceback.print_exc()
                print(
                    f"Failed to load examples with video: {name_sample}. "
                    f"Will randomly sample an example as a replacement."
                )
                index = random.randint(0, len(self) - 1)
                continue

            break

        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return name_sample, pose_sample, text, ""

    def load_pose(self, pose_name, rgb_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), "rb"))

        duration = len(pose["scores"])

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))

        tmp = np.array(tmp)

        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)

        skeletons = pose["keypoints"]
        confs = pose["scores"]
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
        return f"#total {len(self)}"
