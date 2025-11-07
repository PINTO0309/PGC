#!/usr/bin/env python3
"""
Utility script for building pointing/not-pointing datasets from Annot_List.txt.

The script reads the CSV annotation file, copies the requested frames into
chunked folders (1,000 images per sub-directory), generates a labels CSV, prints
class counts, and emits a pie chart visualizing the class balance.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt

Range = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate pointing/not_pointing datasets from Annot_List.txt."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=script_root,
        help="Base directory for relative paths (default: script directory).",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default="data/Annot_List.txt",
        help="Path to the annotation CSV.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="data/images",
        help="Root directory that contains per-video image folders.",
    )
    parser.add_argument(
        "--pointing-dir",
        type=str,
        default="data/dataset/pointing",
        help="Destination for pointing class images.",
    )
    parser.add_argument(
        "--not-pointing-dir",
        type=str,
        default="data/dataset/not_pointing",
        help="Destination for not-pointing class images.",
    )
    parser.add_argument(
        "--labels-output",
        type=str,
        default="data/dataset/labels.txt",
        help="Output CSV that maps image paths to class IDs.",
    )
    parser.add_argument(
        "--chart-output",
        type=str,
        default="data/dataset/class_balance.png",
        help="Pie chart file showing the class distribution.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of images to place in each sub-folder.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Remove pointing/not_pointing directories before copying.",
    )
    parser.add_argument(
        "--skip-id",
        type=int,
        action="append",
        default=None,
        help="Annotation IDs to ignore entirely (default: 1).",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer.")
    return args


def resolve(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_annotations(annotation_path: Path, skip_ids: Set[int]) -> Dict[str, Dict[str, List[Range]]]:
    annotations: Dict[str, Dict[str, List[Range]]] = defaultdict(
        lambda: {"pointing": [], "non_pointing": [], "skip": []}
    )
    with annotation_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"video", "label", "id", "t_start", "t_end", "frames"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation file missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            video = row["video"].strip()
            try:
                label_id = int(row["id"])
                t_start = int(row["t_start"])
                t_end = int(row["t_end"])
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value in row: {row}") from exc
            if label_id in skip_ids:
                annotations[video]["skip"].append((t_start, t_end))
            elif label_id == 2:
                annotations[video]["pointing"].append((t_start, t_end))
            else:
                annotations[video]["non_pointing"].append((t_start, t_end))
    return annotations


def iter_range_frames(ranges: Sequence[Range]) -> Iterable[int]:
    for start, end in ranges:
        if end < start:
            continue
        for frame in range(start, end + 1):
            yield frame


def list_frame_numbers(video_dir: Path, video_name: str) -> List[int]:
    frames: List[int] = []
    pattern = f"{video_name}_"
    for image_path in sorted(video_dir.glob("*.jpg")):
        stem = image_path.stem
        if not stem.startswith(pattern):
            continue
        _, _, suffix = stem.rpartition("_")
        if not suffix.isdigit():
            continue
        frames.append(int(suffix))
    return frames


def ensure_output_dir(path: Path, reset: bool) -> None:
    if reset and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_frames(
    video: str,
    frames: Sequence[int],
    video_dir: Path,
    dest_root: Path,
    chunk_size: int,
    counter: int,
    labels: List[Tuple[str, int]],
    class_id: int,
    base_root: Path,
) -> Tuple[int, int]:
    missing = 0
    for frame in frames:
        src = video_dir / f"{video}_{frame:06d}.jpg"
        if not src.exists():
            missing += 1
            continue
        chunk_dir = dest_root / f"{counter // chunk_size:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        dst = chunk_dir / src.name
        shutil.copy2(src, dst)
        labels.append((format_label_path(dst, base_root), class_id))
        counter += 1
    return counter, missing


def format_label_path(path: Path, base_root: Path) -> str:
    try:
        return path.resolve().relative_to(base_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def save_labels(labels_path: Path, rows: Sequence[Tuple[str, int]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "classid"])
        writer.writerows(rows)


def save_pie_chart(chart_path: Path, pointing_count: int, not_pointing_count: int) -> None:
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.pie(
        [not_pointing_count, pointing_count],
        labels=["not_pointing (0)", "pointing (1)"],
        autopct="%.1f%%",
        startangle=90,
    )
    plt.title("Pointing vs Not Pointing")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()


def main() -> int:
    args = parse_args()
    base_root = args.root.resolve()
    annotation_path = resolve(args.annotation, base_root)
    images_root = resolve(args.images_root, base_root)
    pointing_root = resolve(args.pointing_dir, base_root)
    not_pointing_root = resolve(args.not_pointing_dir, base_root)
    labels_output = resolve(args.labels_output, base_root)
    chart_output = resolve(args.chart_output, base_root)
    skip_ids = set(args.skip_id) if args.skip_id else {1}

    if not annotation_path.exists():
        print(f"Annotation file not found: {annotation_path}", file=sys.stderr)
        return 1
    if not images_root.exists():
        print(f"Images root not found: {images_root}", file=sys.stderr)
        return 1

    annotations = load_annotations(annotation_path, skip_ids)
    if not annotations:
        print("No annotations found after applying skip filters.", file=sys.stderr)
        return 1

    ensure_output_dir(pointing_root, args.reset_output)
    ensure_output_dir(not_pointing_root, args.reset_output)

    pointing_counter = 0
    not_pointing_counter = 0
    missing_files = 0
    labels: List[Tuple[str, int]] = []

    for video in sorted(annotations.keys()):
        video_dir = images_root / video
        if not video_dir.is_dir():
            print(f"[WARN] Missing video directory: {video_dir}")
            continue

        available_frames = list_frame_numbers(video_dir, video)
        if not available_frames:
            print(f"[WARN] No frames found in {video_dir}")
            continue

        available_set = set(available_frames)
        spans = annotations[video]

        pointing_frames = sorted(
            {frame for frame in iter_range_frames(spans["pointing"]) if frame in available_set}
        )
        skip_frames = {frame for frame in iter_range_frames(spans["skip"]) if frame in available_set}
        labeled_non_pointing = {
            frame for frame in iter_range_frames(spans["non_pointing"]) if frame in available_set
        }
        inferred_non_pointing = available_set - set(pointing_frames) - skip_frames
        not_pointing_frames = sorted(labeled_non_pointing | inferred_non_pointing)

        pointing_counter, missing = copy_frames(
            video,
            pointing_frames,
            video_dir,
            pointing_root,
            args.chunk_size,
            pointing_counter,
            labels,
            1,
            base_root,
        )
        missing_files += missing

        not_pointing_counter, missing = copy_frames(
            video,
            not_pointing_frames,
            video_dir,
            not_pointing_root,
            args.chunk_size,
            not_pointing_counter,
            labels,
            0,
            base_root,
        )
        missing_files += missing

        print(
            f"[{video}] pointing={len(pointing_frames)} not_pointing={len(not_pointing_frames)} "
            f"skipped={len(skip_frames)}"
        )

    save_labels(labels_output, labels)
    save_pie_chart(chart_output, pointing_counter, not_pointing_counter)

    print("\nCopy summary")
    print(f"pointing (class 1): {pointing_counter}")
    print(f"not_pointing (class 0): {not_pointing_counter}")
    if missing_files:
        print(f"[WARN] Missing source frames: {missing_files}")
    print(f"Labels CSV: {labels_output}")
    print(f"Pie chart: {chart_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
