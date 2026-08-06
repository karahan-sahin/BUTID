"""Convert MediaPipe pose .h5 files into dense PyTorch tensor files.

Why
---
The .h5 layout stores one HDF5 group per frame, each holding four tiny
datasets (body / left / right / face). Reading one training clip therefore
costs O(frames * 4) HDF5 metadata lookups plus O(frames * 4) tiny reads, which
is what makes NewDataset's IO dominate the step time. It also stores every
landmark in float64, while training only ever uses 69 of the 543 landmarks
(BODY_IDXS_MP: 9, hands: 21 + 21, FACE_IDXS_MP: 18).

This script rewrites each video as a single .pt holding one contiguous
[T, J, 3] tensor per part, pre-sliced to exactly those joints and stored in
float32. A clip read becomes one gather over a memory-mapped tensor: one file
open, one seek per part, ~16x fewer bytes off disk.

Usage
-----
    python converter.py convert --pose-dir /path/to/h5 --out-dir /path/to/pt
    python converter.py convert --pose-dir ... --out-dir ... --workers 16
    python converter.py verify  --pose-dir ... --out-dir ...
    python converter.py bench   --pose-dir ... --out-dir ...

Loading the result
------------------
Point NewDataset at the converted directory and pass use_pt=True:

    NewDataset(csv_dir=..., pose_dir="/path/to/pt", args=args, phase="train", use_pt=True)

or, from train.py, `--pose-dir /path/to/pt --use-pt`. The .h5 path is
unchanged and stays the default.

The reader side (`open_pose_pt`, `window_frames`, `read_pose_window_pt`,
`load_part_mp_pre`) lives in src/dataset.py next to the .h5 reader it mirrors;
this script imports it from there so the written layout and the reader cannot
drift apart.
"""

import argparse
import os
import random
import sys
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# the reader side lives in src.dataset, which is where training consumes it;
# importing it from there keeps the written layout and the reader in lockstep
from src.dataset import (  # noqa: E402
    PART_SPEC,
    PARTS,
    POSE_PT_FORMAT,
    TARGET_FPS,  # noqa: F401  (re-exported for scripts that import converter)
    load_part_mp_pre,
    open_pose_pt,
    read_pose_window_pt,
    window_frames,
)

DTYPES = {"float32": torch.float32, "float16": torch.float16}


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------


def convert_file(h5_path, out_path, dtype="float32", overwrite=False):
    """Pack one video's .h5 into a single .pt.

    Returns (status, odd_shaped), where odd_shaped counts landmark arrays whose
    length was neither 0 nor the expected joint count.
    """
    h5_path, out_path = Path(h5_path), Path(out_path)

    if out_path.exists() and not overwrite:
        return "skipped", 0

    with h5py.File(h5_path, "r") as f:
        if "keypoints" not in f:
            return "no-keypoints", 0

        keypoints_grp = f["keypoints"]

        # sort numerically, not lexically: frame_%04d only sorts correctly as
        # text while a video stays under 10000 frames
        names = sorted(keypoints_grp.keys(), key=lambda n: int(n.split("_")[1]))
        if not names:
            return "empty", 0

        frame_ids = np.array([int(n.split("_")[1]) for n in names], dtype=np.int64)
        num_frames = len(names)

        np_dtype = np.float16 if dtype == "float16" else np.float32
        parts = {
            part: np.zeros((num_frames, len(idxs), 3), dtype=np_dtype)
            for part, (_, idxs, _) in PART_SPEC.items()
        }
        # frames where mediapipe actually produced a detection; missing frames
        # stay zero, which is what load_part_mp substitutes for them anyway
        present = {part: np.zeros(num_frames, dtype=bool) for part in PARTS}

        odd_shaped = 0

        for row, name in enumerate(names):
            frame_grp = keypoints_grp[name]

            for part, (ds_name, idxs, num_joints) in PART_SPEC.items():
                # a part missing from the frame, or stored as a 0-length array,
                # stays zero — exactly what load_part_mp's null_default gives it
                ds = frame_grp.get(ds_name)
                if ds is None or ds.shape[0] == 0:
                    continue
                if ds.shape[0] != num_joints:
                    # unexpected landmark count — treat as a missing detection
                    # rather than silently mis-indexing
                    odd_shaped += 1
                    continue

                parts[part][row] = ds[:][idxs]
                present[part][row] = True

    first_frame = int(frame_ids[0])
    contiguous = bool(frame_ids[-1] - first_frame == num_frames - 1)

    payload = {
        "format": POSE_PT_FORMAT,
        "video_id": h5_path.stem,
        "num_frames": num_frames,
        "first_frame": first_frame,
        # frames are contiguous in every file seen so far; keep the ids anyway
        # so gaps stay representable, and let the reader take the fast path
        "contiguous": contiguous,
        "frame_ids": torch.from_numpy(frame_ids),
        "parts": {part: torch.from_numpy(arr) for part, arr in parts.items()},
        "present": {part: torch.from_numpy(arr) for part, arr in present.items()},
        # the joint sets baked into `parts`; the reader asserts these still
        # match src.dataset, so re-slicing is caught instead of silently wrong
        "joint_index": {part: list(idxs) for part, (_, idxs, _) in PART_SPEC.items()},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # write-then-rename so an interrupted job never leaves a half-written .pt.
    # The temp name carries pid + a random suffix so two converter processes
    # sharing an --out-dir (e.g. one slurm task per shard) cannot collide on it;
    # os.replace is atomic, so whichever finishes last simply wins.
    tmp_path = out_path.with_suffix(f".pt.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return "converted", odd_shaped


def _convert_worker(job):
    h5_path, out_path, dtype, overwrite = job
    try:
        status, odd_shaped = convert_file(h5_path, out_path, dtype, overwrite)
        return h5_path, status, odd_shaped, None
    except Exception as e:  # keep one bad file from killing the whole run
        return h5_path, "failed", 0, f"{type(e).__name__}: {e}"


def shard_files(files, shard_id=0, num_shards=1):
    """Deterministic slice of `files` for one worker of a slurm array.

    Videos differ in size by ~70x, so round-robin over a size-sorted list
    instead of a contiguous split — otherwise one task gets all the big files.
    Every task sees the same ordering, so the shards never overlap.
    """
    if num_shards <= 1:
        return list(files)

    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id {shard_id} out of range for {num_shards} shards")

    ordered = sorted(files, key=lambda p: (-p.stat().st_size, p.name))
    return ordered[shard_id::num_shards]


def convert_dir(
    pose_dir,
    out_dir,
    dtype="float32",
    workers=1,
    overwrite=False,
    limit=0,
    shard_id=0,
    num_shards=1,
):
    pose_dir, out_dir = Path(pose_dir), Path(out_dir)

    h5_files = sorted(pose_dir.glob("*.h5"))
    if limit:
        h5_files = h5_files[:limit]
    if not h5_files:
        print(f"No .h5 files found in {pose_dir}")
        return

    total_files = len(h5_files)
    h5_files = shard_files(h5_files, shard_id, num_shards)
    if num_shards > 1:
        print(f"shard {shard_id}/{num_shards}: {len(h5_files)} of {total_files} files")
    if not h5_files:
        print("nothing to do for this shard")
        return

    jobs = [(p, out_dir / f"{p.stem}.pt", dtype, overwrite) for p in h5_files]

    counts = {}
    failures = []
    odd_total = 0

    def record(result):
        nonlocal odd_total
        h5_path, status, odd_shaped, err = result
        counts[status] = counts.get(status, 0) + 1
        odd_total += odd_shaped
        if err:
            failures.append((h5_path, err))

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            # chunksize=1: videos differ in length by orders of magnitude, so
            # handing out one file at a time keeps the workers balanced
            results = pool.map(_convert_worker, jobs, chunksize=1)
            for result in tqdm(results, total=len(jobs), desc="convert"):
                record(result)
    else:
        for job in tqdm(jobs, desc="convert"):
            record(_convert_worker(job))

    print("\nstatus:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    for h5_path, err in failures[:20]:
        print(f"  [FAILED] {h5_path.name}: {err}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")

    if odd_total:
        print(
            f"  [WARN] {odd_total} landmark arrays had an unexpected joint count "
            "and were stored as missing detections"
        )

    src_bytes = sum(p.stat().st_size for p in h5_files)
    dst_bytes = sum(j[1].stat().st_size for j in jobs if j[1].exists())
    if dst_bytes:
        print(
            f"size: {src_bytes / 2**30:.2f} GiB h5 -> {dst_bytes / 2**30:.2f} GiB pt "
            f"({src_bytes / dst_bytes:.1f}x smaller)"
        )


# ---------------------------------------------------------------------------
# verification — the converted path must reproduce the .h5 path exactly
# ---------------------------------------------------------------------------


def _read_window_h5(h5f, frames):
    """Reference reader — mirrors NewDataset.load_pose's per-frame .h5 access."""
    keypoints_grp = h5f["keypoints"]

    pose = []
    for frame in frames:
        frame_grp = keypoints_grp.get(f"frame_{frame:04d}")
        if frame_grp is None:
            continue

        pose.append(
            {
                "body": frame_grp["pose_landmarks"][:]
                if "pose_landmarks" in frame_grp
                else np.zeros((33, 3), dtype=np.float32),
                "left": frame_grp["left_hand_landmarks"][:]
                if "left_hand_landmarks" in frame_grp
                else np.zeros((21, 3), dtype=np.float32),
                "right": frame_grp["right_hand_landmarks"][:]
                if "right_hand_landmarks" in frame_grp
                else np.zeros((21, 3), dtype=np.float32),
                "face": frame_grp["face_landmarks"][:]
                if "face_landmarks" in frame_grp
                else np.zeros((468, 3), dtype=np.float32),
            }
        )

    return pose or None


def _random_windows(num_frames, count, rng):
    """Plausible (start, end, fps) clip windows over a video of num_frames."""
    windows = []
    for _ in range(count):
        fps = rng.choice([25.0, 29.97, 30.0, 50.0])
        length = rng.randint(20, 300)
        start = rng.randint(0, max(0, num_frames - length - 1))
        windows.append((start, start + length, fps))
    return windows


def verify(
    pose_dir,
    out_dir,
    samples=25,
    start_pad=16,
    end_pad=32,
    seed=0,
    tol=1e-5,
    shard_id=0,
    num_shards=1,
):
    from src.dataset import load_part_mp

    pose_dir, out_dir = Path(pose_dir), Path(out_dir)
    rng = random.Random(seed)

    # shard off the .h5 listing, exactly as convert does, so task i checks the
    # files task i wrote rather than ones another task may still be writing
    h5_files = shard_files(sorted(pose_dir.glob("*.h5")), shard_id, num_shards)
    pt_files = [out_dir / f"{p.stem}.pt" for p in h5_files]
    pt_files = [p for p in pt_files if p.exists()]

    if not pt_files:
        print(f"No .pt files to check in {out_dir} — run `convert` first.")
        return 1

    worst = {part: 0.0 for part in PARTS}
    checked = 0
    length_mismatch = 0

    for pt_path in tqdm(pt_files, desc="verify"):
        h5_path = pose_dir / f"{pt_path.stem}.h5"
        if not h5_path.exists():
            continue

        payload = open_pose_pt(pt_path)

        with h5py.File(h5_path, "r") as h5f:
            for start, end, fps in _random_windows(int(payload["num_frames"]), samples, rng):
                frames = window_frames(start, end, fps, start_pad, end_pad)

                parts = read_pose_window_pt(payload, frames)
                reference = _read_window_h5(h5f, frames)

                if parts is None or reference is None:
                    if (parts is None) != (reference is None):
                        length_mismatch += 1
                    continue

                new = load_part_mp_pre(parts)
                old = load_part_mp(reference)

                for part in PARTS:
                    if new[part].shape != old[part].shape:
                        length_mismatch += 1
                        continue
                    worst[part] = max(
                        worst[part],
                        (new[part].double() - old[part]).abs().max().item(),
                    )

                checked += 1

    print(f"\nchecked {checked} clip windows across {len(pt_files)} videos")
    print(f"{'part':<8}{'max|diff| vs .h5 + load_part_mp':>34}")
    for part in PARTS:
        print(f"{part:<8}{worst[part]:>34.3e}")

    if length_mismatch:
        print(f"\n[FAIL] {length_mismatch} windows differed in length")

    ok = max(worst.values()) <= tol and not length_mismatch
    print(f"\n{'[OK]' if ok else '[FAIL]'} tolerance {tol:g}")

    return 0 if ok else 1


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def bench(pose_dir, out_dir, samples=200, start_pad=16, end_pad=32, seed=0, mmap=True):
    from src.dataset import load_part_mp

    pose_dir, out_dir = Path(pose_dir), Path(out_dir)
    rng = random.Random(seed)

    pt_files = sorted(out_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files in {out_dir} — run `convert` first.")
        return 1

    windows = []
    for pt_path in pt_files:
        h5_path = pose_dir / f"{pt_path.stem}.h5"
        if not h5_path.exists():
            continue
        payload = open_pose_pt(pt_path)
        for w in _random_windows(int(payload["num_frames"]), max(1, samples // len(pt_files)), rng):
            windows.append((pt_path, h5_path, w))

    # h5: reopen per clip, as a worker does when its small file cache misses
    t0 = time.perf_counter()
    for _, h5_path, (start, end, fps) in windows:
        frames = window_frames(start, end, fps, start_pad, end_pad)
        with h5py.File(h5_path, "r") as h5f:
            pose = _read_window_h5(h5f, frames)
        if pose:
            load_part_mp(pose)
    h5_time = (time.perf_counter() - t0) / len(windows)

    t0 = time.perf_counter()
    for pt_path, _, (start, end, fps) in windows:
        frames = window_frames(start, end, fps, start_pad, end_pad)
        payload = open_pose_pt(pt_path, mmap=mmap)
        parts = read_pose_window_pt(payload, frames)
        if parts is not None:
            load_part_mp_pre(parts)
    pt_time = (time.perf_counter() - t0) / len(windows)

    print(f"\n{len(windows)} clips, read + normalize, per clip:")
    print(f"  .h5 + load_part_mp     {h5_time * 1e3:8.2f} ms")
    print(f"  .pt + load_part_mp_pre {pt_time * 1e3:8.2f} ms")
    print(f"  speedup                {h5_time / pt_time:8.1f}x")
    print("\n(cold-cache and network-filesystem gaps are usually larger than this)")

    return 0


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--pose-dir", required=True, help="dir holding the pose .h5 files")
        p.add_argument("--out-dir", required=True, help="dir to write the .pt files to")

    def add_shard(p):
        p.add_argument(
            "--shard-id",
            default=0,
            type=int,
            help="which slice of the videos this process handles (slurm array index)",
        )
        p.add_argument(
            "--num-shards",
            default=1,
            type=int,
            help="total number of slices; shards are disjoint and size-balanced",
        )

    p_convert = sub.add_parser("convert", help="convert .h5 files to .pt")
    add_common(p_convert)
    p_convert.add_argument(
        "--workers",
        default=1,
        type=int,
        help="parallel conversion processes; each holds one whole video in RAM "
        "(~830 bytes/frame, so ~0.5 GiB for a 5 GiB .h5)",
    )
    p_convert.add_argument(
        "--dtype",
        default="float32",
        choices=sorted(DTYPES),
        help="storage dtype; float16 halves the bytes read at ~1e-3 coordinate error",
    )
    p_convert.add_argument("--overwrite", action="store_true", help="re-convert existing .pt files")
    p_convert.add_argument("--limit", default=0, type=int, help="convert only the first N videos")
    add_shard(p_convert)

    p_verify = sub.add_parser("verify", help="check converted clips against the .h5 path")
    add_common(p_verify)
    p_verify.add_argument("--samples", default=25, type=int, help="clip windows per video")
    p_verify.add_argument("--start-pad", default=16, type=int)
    p_verify.add_argument("--end-pad", default=32, type=int)
    p_verify.add_argument("--seed", default=0, type=int)
    p_verify.add_argument("--tol", default=1e-5, type=float)
    add_shard(p_verify)

    p_bench = sub.add_parser("bench", help="time .h5 vs .pt clip loading")
    add_common(p_bench)
    p_bench.add_argument("--samples", default=200, type=int, help="total clip windows to time")
    p_bench.add_argument("--start-pad", default=16, type=int)
    p_bench.add_argument("--end-pad", default=32, type=int)
    p_bench.add_argument("--seed", default=0, type=int)
    p_bench.add_argument("--no-mmap", action="store_false", dest="mmap")

    args = parser.parse_args()

    if args.command == "convert":
        convert_dir(
            args.pose_dir,
            args.out_dir,
            dtype=args.dtype,
            workers=args.workers,
            overwrite=args.overwrite,
            limit=args.limit,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
        return 0

    if args.command == "verify":
        return verify(
            args.pose_dir,
            args.out_dir,
            samples=args.samples,
            start_pad=args.start_pad,
            end_pad=args.end_pad,
            seed=args.seed,
            tol=args.tol,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )

    return bench(
        args.pose_dir,
        args.out_dir,
        samples=args.samples,
        start_pad=args.start_pad,
        end_pad=args.end_pad,
        seed=args.seed,
        mmap=args.mmap,
    )


if __name__ == "__main__":
    sys.exit(main())
