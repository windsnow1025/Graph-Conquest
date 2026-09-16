import struct
import numpy as np

from app.config import SAMPLE_FLOATS


def read_samples(path):
    """Read binary training data exported by TypeScript.

    File format:
      - uint32 LE: number of samples
      - N * SAMPLE_FLOATS float32 values
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        data = np.frombuffer(f.read(n * SAMPLE_FLOATS * 4), dtype=np.float32).copy()
    return data.reshape(n, SAMPLE_FLOATS)


def read_multiple(paths):
    """Read and concatenate multiple binary sample files."""
    arrays = [read_samples(p) for p in paths]
    return np.concatenate(arrays, axis=0)
