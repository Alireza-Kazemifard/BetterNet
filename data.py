import os
import numpy as np
import cv2
import tensorflow as tf
from glob import glob

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _pair_by_stem(images, masks):
    img_map = {_stem(p): p for p in images}
    msk_map = {_stem(p): p for p in masks}
    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    paired_images = [img_map[s] for s in common]
    paired_masks = [msk_map[s] for s in common]
    return paired_images, paired_masks, common


def ensure_split_files(dataset_path: str, split: float = 0.1, seed: int = 42):
    """
    Ensures train.txt and val.txt exist inside dataset_path.
    The txt files store ONLY stems (no extensions), like the old repo style.
    """
    train_txt = os.path.join(dataset_path, "train.txt")
    val_txt = os.path.join(dataset_path, "val.txt")

    if os.path.exists(train_txt) and os.path.exists(val_txt):
        return

    img_dir = os.path.join(dataset_path, "images")
    msk_dir = os.path.join(dataset_path, "masks")

    images = sorted([p for p in glob(os.path.join(img_dir, "*")) if p.lower().endswith(VALID_EXTS)])
    masks = sorted([p for p in glob(os.path.join(msk_dir, "*")) if p.lower().endswith(VALID_EXTS)])

    if len(images) == 0 or len(masks) == 0:
        raise FileNotFoundError(f"Empty images/masks in: {dataset_path}")

    paired_images, paired_masks, stems = _pair_by_stem(images, masks)
    if len(stems) == 0:
        raise RuntimeError(f"No matching image/mask stems in: {dataset_path}")

    rng = np.random.RandomState(seed)
    idx = np.arange(len(stems))
    rng.shuffle(idx)

    n_val = int(round(split * len(stems)))
    n_val = max(1, n_val)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_stems = [stems[i] for i in train_idx]
    val_stems = [stems[i] for i in val_idx]

    with open(train_txt, "w") as f:
        f.write("\n".join(train_stems) + "\n")
    with open(val_txt, "w") as f:
        f.write("\n".join(val_stems) + "\n")


def load_file_names(dataset_path: str, file_name: str):
    """
    Reads stems from train.txt/val.txt and resolves them to actual image/mask paths.
    This mimics the old repo behavior but supports mixed extensions.
    """
    img_dir = os.path.join(dataset_path, "images")
    msk_dir = os.path.join(dataset_path, "masks")
    txt_path = os.path.join(dataset_path, file_name)

    with open(txt_path, "r") as f:
        stems = [line.strip() for line in f if line.strip()]

    images = []
    masks = []
    for s in stems:
        # resolve image
        img_candidates = []
        for ext in VALID_EXTS:
            img_candidates += glob(os.path.join(img_dir, s + ext))
            img_candidates += glob(os.path.join(img_dir, s + ext.upper()))
        # resolve mask
        msk_candidates = []
        for ext in VALID_EXTS:
            msk_candidates += glob(os.path.join(msk_dir, s + ext))
            msk_candidates += glob(os.path.join(msk_dir, s + ext.upper()))

        if not img_candidates or not msk_candidates:
            # skip silently to avoid crash, but this should not happen if split was created by us
            continue

        images.append(sorted(img_candidates)[0])
        masks.append(sorted(msk_candidates)[0])

    return images, masks


def load_data(dataset_paths, split=0.1, seed: int = 42):
    """
    Paper-style: per-dataset 10% unseen split, then combine & shuffle train parts.
    """
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    train_x, train_y = [], []
    val_x, val_y = [], []

    for dp in dataset_paths:
        ensure_split_files(dp, split=split, seed=seed)
        tx, ty = load_file_names(dp, "train.txt")
        vx, vy = load_file_names(dp, "val.txt")

        train_x.extend(tx)
        train_y.extend(ty)
        val_x.extend(vx)
        val_y.extend(vy)

    # combined shuffle (stable)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(train_x))
    rng.shuffle(idx)
    train_x = [train_x[i] for i in idx]
    train_y = [train_y[i] for i in idx]

    idx = np.arange(len(val_x))
    rng.shuffle(idx)
    val_x = [val_x[i] for i in idx]
    val_y = [val_y[i] for i in idx]

    return (train_x, train_y), (val_x, val_y)


def load_test_data(dataset_path: str, split_mode: str = "test", split: float = 0.1, seed: int = 42):
    """
    split_mode:
      - "test": use val.txt (unseen)
      - "full": use all pairs
    """
    img_dir = os.path.join(dataset_path, "images")
    msk_dir = os.path.join(dataset_path, "masks")

    images = sorted([p for p in glob(os.path.join(img_dir, "*")) if p.lower().endswith(VALID_EXTS)])
    masks = sorted([p for p in glob(os.path.join(msk_dir, "*")) if p.lower().endswith(VALID_EXTS)])
    images, masks, _ = _pair_by_stem(images, masks)

    if split_mode == "full":
        return images, masks

    ensure_split_files(dataset_path, split=split, seed=seed)
    vx, vy = load_file_names(dataset_path, "val.txt")
    return vx, vy


def read_image(path):
    if isinstance(path, (bytes, bytearray)):
        path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if x is None:
        raise FileNotFoundError(path)

    # cv2 -> RGB
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    # Lanczos4 resize
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)

    # IMPORTANT: normalize to [0,1] (matches paper & old test)
    x = x.astype(np.float32) / 255.0
    return x


def read_mask(path):
    if isinstance(path, (bytes, bytearray)):
        path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise FileNotFoundError(path)

    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    x = x.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)
    x = (x > 0.5).astype(np.float32)
    return x


def tf_parse(x, y):
    def _parse(xp, yp):
        xi = read_image(xp)
        yi = read_mask(yp)
        return xi, yi

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    y.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    return x, y


def tf_dataset(X, Y, batch_size=8, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)

    if augment:
        # simple flips (paper doesn't explicitly mention aug; keep minimal)
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), tf.image.random_flip_left_right(y)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_flip_up_down(x), tf.image.random_flip_up_down(y)),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
