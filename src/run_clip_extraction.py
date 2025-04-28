import os
import cv2
import json
import pandas as pd
from tqdm import tqdm

def extract_clips(
    video_id,
    video_dir,
    clip_id,
    save_dir,
    start_time,
    end_time,
    bbox,
):
    
    video_path = os.path.join(video_dir, video_id + ".mp4")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video {video_path} not found")
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip_bbox = bbox[video_id]["bbox"]
    
    # Calculate the start and end frames
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Set the video capture to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # Create a VideoWriter object to save the cropped video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        os.path.join(save_dir, clip_id + ".mp4"),
        fourcc,
        fps,
        (clip_bbox["bounding_box_x"][1] - clip_bbox["bounding_box_x"][0], clip_bbox["bounding_box_y"][1] - clip_bbox["bounding_box_y"][0]),
    )
    
    # Read and write frames until the end frame is reached
    progress_bar = tqdm(total=end_frame - start_frame, desc="Extracting clips")
    while cap.isOpened():
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[
            clip_bbox["bounding_box_y"][0] : clip_bbox["bounding_box_y"][1],
            clip_bbox["bounding_box_x"][0] : clip_bbox["bounding_box_x"][1],
        ]

        # Write the cropped frame to the output video
        out.write(cropped_frame)
        progress_bar.update(1)
        
    cap.release()
    out.release()
    progress_bar.close()
    

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

def run_clip_all_clips(clip_df, video_dir, save_dir, bbox):
    print('*'* 20)
    print(clip_df)
    print('*'* 20)
    
    for _, row in tqdm(clip_df.iterrows(), desc="Cropping clips"):
        extract_clips(
            row["video_id"],
            video_dir,
            row["clip_id"],
            save_dir,
            row["start_in_seconds"],
            row["end_in_seconds"],
            bbox
        )

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Crop all videos in the dataset")
    parser.add_argument('--annotation_file', type=str, help="Path to the annotation file")
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

    if args.annotation_file:
        annotation_df =  pd.read_csv(args.annotation_file)
        annotation_df = annotation_df[annotation_df["video_id"].isin(video_ids)]
        # Split the dataframe into chunks
        annotation_df = [
            annotation_df.iloc[i::num_workers] for i in range(num_workers)
        ]

    os.makedirs(args.save_dir, exist_ok=True)

    threads = []
    for i in range(num_workers):
        if args.annotation_file:
            # Split the dataframe into chunk with chunk number of rows
            chunk = annotation_df[i]
            # Create a thread for each chunk    
            t = threading.Thread(
                target=run_clip_all_clips,
                args=(
                    chunk,
                    args.video_dir,
                    args.save_dir,
                    bbox,
                ),
            )
            threads.append(t)
            t.start()
        else:
            t = threading.Thread(
                target=run_clip_all_videos,
                args=(video_ids[i], args.video_dir, args.save_dir, bbox),
            )
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    print("Done")
