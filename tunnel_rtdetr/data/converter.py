
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from tunnel_rtdetr.utils import DATASETS_DIR, LOGGER, TQDM, YAML
from tunnel_rtdetr.utils.checks import check_file


def coco91_to_coco80_class() -> list[int]:
    """Convert 91-index COCO class IDs to 80-index COCO class IDs.

    Returns:
        (list[int]): A list of 91 class IDs where the index represents the 80-index class ID and the value is the
            corresponding 91-index class ID.
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def coco80_to_coco91_class() -> list[int]:
    r"""Convert 80-index (val2014) to 91-index (paper).

    Returns:
        (list[int]): A list of 80 class IDs where each value is the corresponding 91-index class ID.

    Examples:
        >>> import numpy as np
        >>> a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        >>> b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")

        Convert the darknet to COCO format
        >>> x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]

        Convert the COCO to darknet format
        >>> x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]

    References:
        https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]








def min_index(arr1: np.ndarray, arr2: np.ndarray):
    """Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        idx1 (int): Index of the point in arr1 with the shortest distance.
        idx2 (int): Index of the point in arr2 with the shortest distance.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments: list[list]):
    """merge multi segment. RT-DETR detection workflow."""
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s








async def convert_ndjson_to_yolo(ndjson_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert NDJSON dataset format to Tunnel RT-DETR YOLO dataset structure.

    This function converts datasets stored in NDJSON (Newline Delimited JSON) format to the standard YOLO format. For
    detection/segmentation/pose/obb tasks, it creates separate directories for images and labels. For classification
    tasks, it creates the ImageNet-style {split}/{class_name}/ folder structure. It supports parallel processing for
    efficient conversion of large datasets and can download images from URLs.

    The NDJSON format consists of:
    - First line: Dataset metadata with class names, task type, and configuration
    - Subsequent lines: Individual image records with annotations and optional URLs

    Args:
        ndjson_path (Union[str, Path]): Path to the input NDJSON file containing dataset information.
        output_path (Optional[Union[str, Path]], optional): Directory where the converted YOLO dataset will be saved. If
            None, uses the parent directory of the NDJSON file. Defaults to None.

    Returns:
        (Path): Path to the generated data.yaml file (detection) or dataset directory (classification).

    Examples:
        Convert a local NDJSON file:
        >>> yaml_path = convert_ndjson_to_yolo("dataset.ndjson")
        >>> print(f"Dataset converted to: {yaml_path}")

        Convert with custom output directory:
        >>> yaml_path = convert_ndjson_to_yolo("dataset.ndjson", output_path="./converted_datasets")

        Use with YOLO training
        >>> from tunnel_rtdetr import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> model.train(data="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-ndjson.ndjson")
    """
    from tunnel_rtdetr.utils.checks import check_requirements

    check_requirements("aiohttp")
    import aiohttp

    ndjson_path = Path(check_file(ndjson_path))
    output_path = Path(output_path or DATASETS_DIR)
    with open(ndjson_path) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    dataset_record, image_records = lines[0], lines[1:]
    dataset_dir = output_path / ndjson_path.stem
    splits = {record["split"] for record in image_records}

    # Check if this is a classification dataset
    is_classification = dataset_record.get("task") == "classify"
    class_names = {int(k): v for k, v in dataset_record.get("class_names", {}).items()}
    len(class_names)

    # Validate required fields before downloading images
    task = dataset_record.get("task", "detect")
    if not is_classification:
        if "train" not in splits:
            raise ValueError(f"Dataset missing required 'train' split. Found splits: {sorted(splits)}")
        if "val" not in splits and "test" not in splits:
            raise ValueError(f"Dataset missing required 'val' split. Found splits: {sorted(splits)}")
    if task == "pose" and "kpt_shape" not in dataset_record:
        raise ValueError("Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose/")

    # Create base directories
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = None

    if not is_classification:
        # Detection/segmentation/pose/obb: prepare YAML and create base structure
        data_yaml = dict(dataset_record)
        data_yaml["names"] = class_names
        data_yaml.pop("class_names", None)
        data_yaml.pop("type", None)  # Remove NDJSON-specific fields
        for split in sorted(splits):
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            data_yaml[split] = f"images/{split}"

    async def process_record(session, semaphore, record):
        """Process single image record with async session."""
        async with semaphore:
            split, original_name = record["split"], record["file"]
            annotations = record.get("annotations", {})

            if is_classification:
                # Classification: place image in {split}/{class_name}/ folder
                class_ids = annotations.get("classification", [])
                class_id = class_ids[0] if class_ids else 0
                class_name = class_names.get(class_id, str(class_id))
                image_path = dataset_dir / split / class_name / original_name
            else:
                # Detection: write label file and place image in images/{split}/
                image_path = dataset_dir / "images" / split / original_name
                label_path = dataset_dir / "labels" / split / f"{Path(original_name).stem}.txt"
                lines_to_write = []
                for key in annotations.keys():
                    lines_to_write = [" ".join(map(str, item)) for item in annotations[key]]
                    break
                label_path.write_text("\n".join(lines_to_write) + "\n" if lines_to_write else "")

            # Download image if URL provided and file doesn't exist
            if http_url := record.get("url"):
                if not image_path.exists():
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    # Retry with exponential backoff (3 attempts: 0s, 2s, 4s delays)
                    for attempt in range(3):
                        try:
                            async with session.get(http_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                                response.raise_for_status()
                                image_path.write_bytes(await response.read())
                            return True
                        except Exception as e:
                            if attempt < 2:  # Don't sleep after last attempt
                                await asyncio.sleep(2**attempt)  # 1s, 2s backoff
                            else:
                                LOGGER.warning(f"Failed to download {http_url} after 3 attempts: {e}")
                                return False
            return True

    # Process all images with async downloads (limit connections for small datasets)
    semaphore = asyncio.Semaphore(min(128, len(image_records)))
    async with aiohttp.ClientSession() as session:
        pbar = TQDM(
            total=len(image_records),
            desc=f"Converting {ndjson_path.name} → {dataset_dir} ({len(image_records)} images)",
        )

        async def tracked_process(record):
            result = await process_record(session, semaphore, record)
            pbar.update(1)
            return result

        results = await asyncio.gather(*[tracked_process(record) for record in image_records])
        pbar.close()

    # Validate images were downloaded successfully
    success_count = sum(1 for r in results if r)
    if success_count == 0:
        raise RuntimeError(f"Failed to download any images from {ndjson_path}. Check network connection and URLs.")
    if success_count < len(image_records):
        LOGGER.warning(f"Downloaded {success_count}/{len(image_records)} images from {ndjson_path}")

    if is_classification:
        # Classification: return dataset directory (check_cls_dataset expects a directory path)
        return dataset_dir
    else:
        # Detection: write data.yaml and return its path
        yaml_path = dataset_dir / "data.yaml"
        YAML.save(yaml_path, data_yaml)
        return yaml_path
