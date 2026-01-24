import cv2
import h5py
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm.auto import tqdm

# Only select the landmarks that are needed
FACE_LANDMARKS = [
    0,
    4,
    13,
    14,
    17,
    33,
    37,
    39,
    46,
    52,
    55,
    61,
    64,
    81,
    82,
    93,
    133,
    151,
    152,
    159,
    172,
    178,
    181,
    263,
    269,
    276,
    282,
    285,
    291,
    294,
    311,
    323,
    362,
    386,
    397,
    # 468, 473
]

import warnings

warnings.filterwarnings("ignore")

# Initialize MediaPipe Pose.
mp_holistic = mp.solutions.holistic


def get_landmarks(
    video_path,
    start_time=None,
    end_time=None,
    end_offset=3,
    bounding_box=None,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    float_precision=3,
    detect_people=False,
):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Initialize video capture.
    cap = cv2.VideoCapture(video_path)
    print(f" ** ** Reading video: {video_path} **")

    start_frame = int(start_time * cap.get(cv2.CAP_PROP_FPS))
    end_frame = int(end_time * cap.get(cv2.CAP_PROP_FPS))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if end_frame > 0:
        end_frame = min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Add end offset to the end frame
    end_frame += int(end_offset * cap.get(cv2.CAP_PROP_FPS))

    end = end_frame - start_frame

    # Create a list to store pose landmarks.
    pose_landmarks = []
    left_hand_landmarks = []
    right_hand_landmarks = []
    face_landmarks = []

    cap_idx = 0

    progress_bar = tqdm(total=end, desc="Processing video frames for clip", leave=False)
    with mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=model_complexity,
    ) as holistic:

        while cap.isOpened():

            if cap_idx >= end:
                break

            cap_idx += 1
            ret, frame = cap.read()
            if not ret:
                break

            if bounding_box:
                image = frame[
                    bounding_box["bounding_box_y"][0] : bounding_box["bounding_box_y"][
                        1
                    ],
                    bounding_box["bounding_box_x"][0] : bounding_box["bounding_box_x"][
                        1
                    ],
                ]

            # Process the image and extract pose landmarks.
            results = holistic.process(image)

            pose_landmark, left_hand_landmark, right_hand_landmark, face_landmark = (
                None,
                None,
                None,
                None,
            )
            if results.pose_landmarks:
                # Extract landmarks and convert them to a list of tuples.
                pose_landmark = [
                    (lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark
                ]
            if results.left_hand_landmarks:
                # Extract landmarks and convert them to a list of tuples.
                left_hand_landmark = [
                    (lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark
                ]
            if results.right_hand_landmarks:
                # Extract landmarks and convert them to a list of tuples.
                right_hand_landmark = [
                    (lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark
                ]
            if results.face_landmarks:
                # Extract landmarks and convert them to a list of tuples.
                face_landmark = [
                    (lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark
                ]

            # Append the landmarks to the list.
            pose_landmarks.append(
                np.array(pose_landmark) if pose_landmark else np.zeros((33, 3))
            )
            left_hand_landmarks.append(
                np.array(left_hand_landmark)
                if left_hand_landmark
                else np.zeros((21, 3))
            )
            right_hand_landmarks.append(
                np.array(right_hand_landmark)
                if right_hand_landmark
                else np.zeros((21, 3))
            )
            face_landmarks.append(
                np.array(face_landmark) if face_landmark else np.zeros((468, 3))
            )

            progress_bar.update(1)

    pose_landmarks = np.array(pose_landmarks)
    left_hand_landmarks = np.array(left_hand_landmarks)
    right_hand_landmarks = np.array(right_hand_landmarks)
    face_landmarks = np.array(face_landmarks)

    # Round float values to 3 decimal places.
    pose_landmarks = np.round(pose_landmarks, args.float_precision)
    left_hand_landmarks = np.round(left_hand_landmarks, args.float_precision)
    right_hand_landmarks = np.round(right_hand_landmarks, args.float_precision)
    face_landmarks = np.round(face_landmarks, args.float_precision)

    # Select only the landmarks
    face_landmarks = face_landmarks[:, FACE_LANDMARKS]

    print(f" ** ** Pose landmarks shape: {pose_landmarks.shape}")
    print(f" ** ** Left hand landmarks shape: {left_hand_landmarks.shape}")
    print(f" ** ** Right hand landmarks shape: {right_hand_landmarks.shape}")
    print(f" ** ** Face landmarks shape: {face_landmarks.shape}")

    # Release the video capture and close windows.
    cap.release()
    cv2.destroyAllWindows()

    return pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks


def write_pose_landmarks_to_hdf5(
    clip_id: str,
    sentence: str,
    context: list[str],
    pose_landmarks: np.ndarray,
    left_hand_landmarks: np.ndarray,
    right_hand_landmarks: np.ndarray,
    face_landmarks: np.ndarray,
    output_file: str,
):

    print(f"Writing pose landmarks to HDF5 file: {clip_id}")
    print(f"Pose landmarks shape: {pose_landmarks.shape}")
    print(f"Left hand landmarks shape: {left_hand_landmarks.shape}")
    print(f"Right hand landmarks shape: {right_hand_landmarks.shape}")
    print(f"Face landmarks shape: {face_landmarks.shape}")

    # Create an HDF5 file to store the pose landmarks.
    with h5py.File(output_file, "a") as f:

        if clip_id in f:
            print(f"Clip ID already exists: {clip_id}")

            # Check if the pose landmarks are already written.
            if "joints" in f[clip_id]:
                print(f"Pose landmarks already written: {clip_id}")
                return

        else:
            # Create a group to store the clip information.
            f.create_group(clip_id)

            if not "sentence" in f[clip_id]:
                f[clip_id].create_dataset("sentence", data=sentence)
            if not "context" in f[clip_id]:
                f[clip_id].create_dataset("context", data=context)

        # Create a group to store the landmarks named 'joints'
        f[clip_id].create_group("joints")
        f[f"{clip_id}/joints"].create_dataset("pose", data=pose_landmarks)
        f[f"{clip_id}/joints"].create_dataset("left_hand", data=left_hand_landmarks)
        f[f"{clip_id}/joints"].create_dataset("right_hand", data=right_hand_landmarks)
        f[f"{clip_id}/joints"].create_dataset("face", data=face_landmarks)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Get pose landmarks from a video.")
    parser.add_argument(
        "--video_path",
        type=str,
        default="data/clips/1.mp4",
        help="Path to the video file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/pose_landmarks.hdf5",
        help="Path to the output HDF5 file.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/subtitles_pad_2seconds.csv",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--model_complexity",
        type=int,
        default=2,
        help="Model complexity.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.1,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.1,
        help="Minimum tracking confidence.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=4,
        help="Number of chunks to process the dataset.",
    )
    parser.add_argument(
        "--end_offset", type=int, default=3, help="End offset in seconds."
    )
    parser.add_argument(
        "--float_precision", type=int, default=3, help="Float precision."
    )
    parser.add_argument(
        "--bbox_path",
        type=str,
        default="data/bbox.json",
        help="Path to the bounding box json file.",
    )
    parser.add_argument("--rank", type=int, default=0, help="Rank of the process.")

    return parser.parse_args()


if __name__ == "__main__":

    import os

    args = parse_args()

    import json

    with open(args.bbox_path, "r") as f:
        bbox = json.load(f)

    def process_video(row, output_file):

        video_id = row.video_id
        clip_id = row.clip_id
        sentence = row.text
        context = []
        video_path = os.path.join(args.video_path, f"{video_id}.mp4")
        start = row.start_in_seconds
        end = row.end_in_seconds

        # Check if the video file exists.
        if video_path.split("/")[-1] not in os.listdir(args.video_path):
            print(f"Video file not found: {video_path}")
            return

        def check_clip_id(clip_id, output_file):
            with h5py.File(output_file, "a") as f:
                if clip_id in f:
                    print(f"Clip ID already exists: {clip_id}")
                    return True
            return False

        if check_clip_id(clip_id, output_file):
            return

        # Get the pose landmarks.
        (pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks) = (
            get_landmarks(
                video_path,
                start_time=start,
                end_time=end,
                end_offset=args.end_offset,
                bounding_box=bbox[video_id]["bbox"],
                model_complexity=args.model_complexity,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                float_precision=args.float_precision,
            )
        )

        # If all pose landmarks are zeros, return None.
        if np.all(pose_landmarks == 0):
            print(f"Pose landmarks are all zeros: {clip_id}")
            return

        # If both left and right hand landmarks are zeros, return None.
        if np.all(left_hand_landmarks == 0) and np.all(right_hand_landmarks == 0):
            print(f"Hand landmarks are all zeros: {clip_id}")
            return

        # Write the pose landmarks to an HDF5 file.
        write_pose_landmarks_to_hdf5(
            clip_id=clip_id,
            sentence=sentence,
            context=context,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks,
            output_file=output_file,
        )

    # Load the dataset
    df = pd.read_csv(args.csv_path)
    print(f" ** Number of samples: {len(df)}")

    # Filter the dataset if cropped videos are not available
    available_ids = [x.split(".")[0] for x in os.listdir(args.video_path) if x]
    df = df[df["video_id"].isin(available_ids)]
    print(f" ** Number of samples after video filtering: {len(df)}")

    # Filter bbox annotations that are not available
    df = df[df["video_id"].isin(bbox.keys())]
    print(f" ** Number of samples after bbox filtering: {len(df)}")

    # Filter the dataset if the clip id is already used
    # df = df[~df['clip_id'].isin(used_ids)]
    # print(f' ** Number of samples after clip filtering: {len(df)}')

    def process_chunk(df, chunk_id):

        # Print video ids in the chunk
        print(f" ** Chunk ID: {chunk_id}")
        print(f" ** Number of samples in the chunk: {len(df)}")
        print(f" ** Video IDs in the chunk: {df['video_id'].unique()}")

        for i, row in df.iterrows():

            video_path = os.path.join(args.video_path, f"{row.video_id}.mp4")

            if not os.path.exists(video_path):
                print(f" ** File not found: {video_path}")
                continue

            print(f" ** Processing video: {video_path}")
            print(f" ** Clip ID: {row.clip_id}")
            print(f" ** Sentence: {row.text}")
            print(f" ** Start time: {row.start_in_seconds}")
            print(f" ** End time: {row.end_in_seconds}")
            print(f"Start:", row.start, "End:", row.end)

            try:
                process_video(row=row, output_file=f"{args.output_file}.{chunk_id}.h5")
            except Exception as e:
                print(f" ** Error processing video: {video_path}")
                print(e)
                continue

    if args.num_chunks > 1:
        # Process the dataset in chunks
        # chunks = [chunk for chunk in np.array_split(df, args.num_chunks)]
        # Create chunks based on the video id
        chunks = []
        unique_video_ids = df["video_id"].unique()

        if args.rank == 0:
            # Process the first half of the dataset
            unique_video_ids = unique_video_ids[: len(unique_video_ids) // 2]
        if args.rank == 1:
            # Process the second half of the dataset
            unique_video_ids = unique_video_ids[len(unique_video_ids) // 2 :]

        for i in range(args.num_chunks):
            chunk = df[df["video_id"].isin(unique_video_ids[i :: args.num_chunks])]
            # Print video ids in the chunk
            print(f" ** Chunk ID: {i}")
            print(f" ** Number of samples in the chunk: {len(chunk)}")
            print(
                f" ** Number of Video IDs in the chunk: {chunk['video_id'].unique().shape[0]}"
            )
            chunks.append(chunk)

        print(f" ** Sum of the chunks: {sum([len(chunk) for chunk in chunks])}")
        print(f" ** Length of the dataset: {len(df)}")

        assert sum([len(chunk) for chunk in chunks]) == len(
            df[df.video_id.isin(unique_video_ids)]
        ), "Chunks are not equal to the dataset."

        # Create threads to process the chunks
        import threading

        threads = []
        for chunk_id, chunk in enumerate(chunks):

            if args.rank == 1:
                chunk_id += args.num_chunks

            print(f"Processing chunk {chunk_id}...")

            t = threading.Thread(target=process_chunk, args=(chunk, chunk_id))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    else:
        print("Processing the dataset in a single chunk.", os.listdir(args.video_path))
        # Process the dataset in a single
        process_chunk(df, 0)

    print("Pose landmarks saved to HDF5 file.")
