import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import CONFIG


def _resolve_np_dtype(dtype_value):
    if dtype_value is None:
        return np.int32
    if isinstance(dtype_value, str):
        return np.dtype(dtype_value)
    return np.dtype(dtype_value)


# =========================================================
# 1️⃣ PRETOKENIZE (Jalankan SEKALI saja sebelum training)
# =========================================================

def pretokenize_corpus(
    corpus_path,
    tokenizer,
    output_path=None,
    dtype=None,
    log_interval=None,
):
    """
    Convert corpus.txt → tokens.bin (int32)
    Hanya dijalankan sekali.
    """

    output_path = output_path or str(getattr(CONFIG, "PRETOKENIZE_OUTPUT_PATH", "tokens.bin"))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dtype = _resolve_np_dtype(dtype or getattr(CONFIG, "PRETOKENIZE_DTYPE", np.int32))
    log_interval = int(log_interval if log_interval is not None else getattr(CONFIG, "PRETOKENIZE_LOG_INTERVAL", 10000))

    print("Starting pretokenization...")
    all_ids = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            ids = tokenizer.encode_ids(line, add_bos=True, add_eos=True)

            all_ids.extend(ids)

            if i % log_interval == 0 and i > 0:
                print(f"Processed {i} lines")

    arr = np.array(all_ids, dtype=dtype)
    arr.tofile(output_path)

    print("Pretokenization finished.")
    print("Saved to:", output_path)
    print("Total tokens:", len(arr))


# =========================================================
# 2️⃣ PACKED DATASET (LLM STANDARD)
# =========================================================

class PackedDataset(Dataset):
    """
    Production-level dataset using memory-mapped token file.
    No padding.
    Auto label shifting.
    """

    def __init__(
        self,
        token_file,
        block_size=None,
        dtype=None,
    ):
        assert os.path.exists(token_file), f"{token_file} not found"

        self.block_size = int(block_size if block_size is not None else getattr(CONFIG, "DATASET_BLOCK_SIZE", 128))
        dtype = _resolve_np_dtype(dtype or getattr(CONFIG, "DATASET_DTYPE", np.int32))
        self.data = np.memmap(token_file, dtype=dtype, mode="r")

        self.total_tokens = len(self.data)
        self.num_blocks = (self.total_tokens - 1) // self.block_size

        print("Loaded token file:", token_file)
        print("Total tokens:", self.total_tokens)
        print("Total blocks:", self.num_blocks)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1

        chunk = self.data[start:end]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return {
            "input_ids": x,
            "labels": y,
        }


# =========================================================
# 3️⃣ SLIDING WINDOW DATASET (Optional)
# =========================================================

class SlidingWindowDataset(Dataset):
    """
    Lebih fleksibel tapi lebih mahal.
    """

    def __init__(
        self,
        token_file,
        block_size=None,
        stride=None,
        dtype=None,
    ):
        self.block_size = int(block_size if block_size is not None else getattr(CONFIG, "DATASET_BLOCK_SIZE", 128))
        self.stride = int(stride if stride is not None else getattr(CONFIG, "SLIDING_STRIDE", 1))
        dtype = _resolve_np_dtype(dtype or getattr(CONFIG, "DATASET_DTYPE", np.int32))
        self.data = np.memmap(token_file, dtype=dtype, mode="r")

        self.total_tokens = len(self.data)
        self.num_samples = max(
            0,
            ((self.total_tokens - self.block_size - 1) // self.stride) + 1,
        )

        print("Sliding Dataset loaded")
        print("Total samples:", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.block_size + 1

        chunk = self.data[start:end]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return {
            "input_ids": x,
            "labels": y,
        }


# =========================================================
# 4️⃣ DATALOADER FACTORY
# =========================================================

def create_dataloader(
    token_file=None,
    block_size=None,
    batch_size=None,
    shuffle=None,
    num_workers=None,
    pin_memory=None,
    persistent_workers=None,
    drop_last=None,
    dataset_type=None,
    stride=None,
    dtype=None,
):
    """
    dataset_type: "packed" or "sliding"
    """

    token_file = token_file or str(getattr(CONFIG, "DATASET_TOKEN_FILE", "tokens.bin"))
    block_size = int(block_size if block_size is not None else getattr(CONFIG, "DATASET_BLOCK_SIZE", 128))
    batch_size = int(batch_size if batch_size is not None else getattr(CONFIG, "DATASET_BATCH_SIZE", 16))
    shuffle = bool(shuffle if shuffle is not None else getattr(CONFIG, "DATASET_SHUFFLE", True))
    num_workers = int(num_workers if num_workers is not None else getattr(CONFIG, "DATASET_NUM_WORKERS", 0))
    pin_memory = bool(pin_memory if pin_memory is not None else getattr(CONFIG, "DATASET_PIN_MEMORY", True))
    persistent_workers = bool(
        persistent_workers if persistent_workers is not None else getattr(CONFIG, "DATASET_PERSISTENT_WORKERS", False)
    )
    drop_last = bool(drop_last if drop_last is not None else getattr(CONFIG, "DATASET_DROP_LAST", False))
    if num_workers == 0:
        persistent_workers = False

    dataset_type = str(dataset_type or getattr(CONFIG, "DATASET_TYPE", "packed")).lower()
    stride = int(stride if stride is not None else getattr(CONFIG, "SLIDING_STRIDE", 1))
    dtype = _resolve_np_dtype(dtype or getattr(CONFIG, "DATASET_DTYPE", np.int32))

    if dataset_type == "packed":
        dataset = PackedDataset(
            token_file=token_file,
            block_size=block_size,
            dtype=dtype,
        )
    elif dataset_type == "sliding":
        dataset = SlidingWindowDataset(
            token_file=token_file,
            block_size=block_size,
            stride=stride,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'packed' or 'sliding'.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    return loader
