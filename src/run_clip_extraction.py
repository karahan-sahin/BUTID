import os
import cv2
import json
from tqdm import tqdm

def crop_video(video_id, video_dir, save_dir, bbox):

    video_fpath = os.path.join(video_dir, video_id + ".mp4")

    if not os.path.exists(video_fpath):
        raise FileNotFoundError(f"Video {video_fpath} not found")

    # Crop bbox from the video and save it crop/clip_id
    cap = cv2.VideoCapture(video_fpath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_bbox = bbox[video_id]["bbox"]

    # Get a sample frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()

    # Crop the frame
    cropped_frame = frame[
        clip_bbox["bounding_box_y"][0] : clip_bbox["bounding_box_y"][1],
        clip_bbox["bounding_box_x"][0] : clip_bbox["bounding_box_x"][1],
    ]

    # Save frames into a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(save_dir, video_id + ".mp4"),
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (cropped_frame.shape[1], cropped_frame.shape[0]),
    )

    tqdm.write(f"Width: {width}, Height: {height}")
    progress_bar = tqdm(
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Cropping video"
    )
    while cap.isOpened():
        try:
            ret, frame = cap.read()

            if not ret:
                break

            out.write(
                frame[
                    clip_bbox["bounding_box_y"][0] : clip_bbox["bounding_box_y"][1],
                    clip_bbox["bounding_box_x"][0] : clip_bbox["bounding_box_x"][1],
                ]
            )
            progress_bar.update(1)
        except Exception as e:
            print(e)
            break

    cap.release()
    out.release()
    progress_bar.close()


def run_clip_all_videos(video_ids, video_dir, save_dir, bbox):
    for video_id in tqdm(video_ids, desc="Cropping videos"):
        crop_video(video_id, video_dir, save_dir, bbox)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Crop all videos in the dataset")
    parser.add_argument("--video_dir", type=str, help="Path to the video directory")
    parser.add_argument("--save_dir", type=str, help="Path to the save directory")
    parser.add_argument("--bbox_file", type=str, help="Path to the bounding box")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers to use"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.bbox_file, "r") as f:
        bbox = json.load(f)

    video_ids = list(bbox.keys())

    # Check if the video directory exists and video_ids are in the directory
    video_ids = [
        video_id
        for video_id in video_ids
        if os.path.exists(os.path.join(args.video_dir, video_id + ".mp4"))
    ]

    import threading

    num_workers = args.num_workers
    video_ids = [video_ids[i::num_workers] for i in range(num_workers)]

    os.makedirs(args.save_dir, exist_ok=True)

    threads = []
    for i in range(num_workers):
        t = threading.Thread(
            target=run_clip_all_videos,
            args=(video_ids[i], args.video_dir, args.save_dir, bbox),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Done")
