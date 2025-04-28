import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def format_timestamp(timestamp, add_to_sc):
    """
    Formats a timestamp to be used in yt-dlp.

    :param timestamp: Timestamp in the format 'hh:mm:ss.sss'.
    :param add_to_sc: Value to add to the seconds.

    :return: Formatted timestamp in the format 'hh:mm:ss'.
    """
    h, m, sc = timestamp.split(":")
    s, ms = sc.split(".")
    s = int(s) + add_to_sc
    return f"{h}:{m}:{s}"

def download_video(video_url, output_file=None):
    """
    Downloads a YouTube video.

    :param video_url: URL of the YouTube video.
    :param output_file: Optional output filename.
    """

    if os.path.exists(output_file) or os.path.exists(output_file.replace('.mp4', '.mkv')) or os.path.exists(output_file.replace('.mp4', '.webm')) or os.path.exists(output_file.replace('.mp4', '_cropped.mp4')):
        print(f"File {output_file} already exists.")
        return

    from pytubefix import YouTube

    yt = YouTube(video_url)
    print(f"Downloading video: {yt.title}")
    # download with no audio
    file_name = output_file.split('/')[-1]
    output_dir = '/'.join(output_file.split('/')[:-1])
    yt.streams.filter(only_video=True, file_extension='mp4').first().download(output_path=output_dir, filename=file_name)


def download_video_section(row, output_file=None, end_time_offset=3):
    """
    Downloads a specific section of a YouTube video.

    :param video_url: URL of the YouTube video.
    :param output_file: Optional output filename.
    :param end_time_offset: Optional offset to add to the end time.
    """

    start_time = format_timestamp(row.start, 0)
    end_time = format_timestamp(row.end, end_time_offset)
    _id = row.video_id
    time_span = f"*{start_time}-{end_time}"

    # os.system(
    #     f'yt-dlp --force-keyframes-at-cuts --recode mp4 -o {output_file} -q --download-sections "{time_span}" -- https://www.youtube.com/watch?v={_id}'
    # )

def crop_video_display(video_path, x_percentage, y_percentage, output_path):
    """
    Crops a video to the display size. The display size is calculated by multiplying the width and height by the
    percentage values provided.

    :param video_path: Path to the video.
    :param x_percentage: Percentage of the width to crop.
    :param y_percentage: Percentage of the height to crop.
    :param output_path: Path to save the cropped video.
    """
    # os.system(
    #     f"ffmpeg -i {video_path} -vf crop=iw*{x_percentage}:ih*{y_percentage}:iw*{1-x_percentage}:ih*{1-y_percentage} {output_path}"
    # )

    # Do this in opencv
    import cv2
    # First check if the file exists
    if os.path.exists(video_path) or os.path.exists(video_path.replace('.mp4', '.mkv')) or os.path.exists(video_path.replace('.mp4', '.webm')):
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return
        
        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the new width and height (bottom right corner)
        w = int(width * x_percentage)
        h = int(height * y_percentage)
        
        # Write the video to a new file
        # Use cv2.VideoWriter_fourcc('F','M','P','4')
        out = cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            cap.get(cv2.CAP_PROP_FPS), 
            (w, h))
        
        while True:
            if not cap.isOpened():
                break
            ret, frame = cap.read()
            if not ret: 
                break
            # Crop the frame (bottom right corner)
            frame = frame[-h:, -w:]

            out.write(frame)

        cap.release()
        out.release()
    else:
        print(f"File not found: {video_path}")

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Download clips from YouTube videos.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/clips",
        help="Directory to save the clips.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=10,
        help="Number of chunks to split the dataset into.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/subtitles_pad_2seconds.csv",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--end_time_offset", type=int, default=3, help="Offset to add to the end time."
    )
    parser.add_argument(
        '--download_whole_video',
        action='store_true',
        help="Download the whole video instead of a section."
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=None,
        help="Local rank for distributed training."
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Load the dataset
    df = pd.read_csv(args.dataset_path)

    # Split dataframe into 10 parts
    chunks = [chunk for chunk in np.array_split(df, args.num_chunks)]

    # Create a directory to save the clips
    os.makedirs(args.save_dir, exist_ok=True)

    def download_videos(df):
        for i, row in tqdm(df.iterrows()):
            # output_file = f"{args.save_dir}/{row.video_id}.{row.start_in_seconds}.{row.end_in_seconds}.mp4"
            output_file = f"{args.save_dir}/{row.video_id}.mp4"
            # cropped_output_file = f"{args.save_dir}/{row.video_id}.{str(row.start_in_seconds).zfill(12)}.{str(row.end_in_seconds).zfill(12)}.mp4"
            cropped_output_file = f"{args.save_dir}/{row.video_id}_cropped.mp4"
            if os.path.exists(cropped_output_file):
                print(f"File {output_file} already exists.")
                continue
            # Download the video section
            # download_video_section(
            #     row,
            #     output_file=output_file,
            #     end_time_offset=args.end_time_offset,
            # )
            download_video(video_url=f'youtube.com/watch?v={row.video_id}',
                           output_file=output_file)
            # # Crop the video to the display size
            # crop_video_display(
            #     video_path=output_file,
            #     x_percentage=0.5,
            #     y_percentage=0.5,
            #     output_path=cropped_output_file
            # )
            # Remove the original file
            try:
                os.remove(output_file)
            except FileNotFoundError:
                print(f"File {output_file} not found.")
                pass

            # Sleep for 1 second to avoid getting banned
            time.sleep(1)

    if args.local_rank is not None:
        # Split the chunks into equal parts
        download_videos(
            chunks[args.local_rank]
        )

    # Create different threads for each chunk
    import threading

    threads = []
    for i, chunk in enumerate(chunks):
        thread = threading.Thread(target=download_videos, args=(chunk,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All clips downloaded successfully!")
