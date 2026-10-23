import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.data_align import unwrap_ls  # noqa: E402


def test_unwrap_ls_applies_one_offset_after_one_wrap():
    wrapped = np.array([359.9, 0.1, 0.2, 1.0], dtype=np.float64)

    result = unwrap_ls(wrapped)

    np.testing.assert_allclose(result, [359.9, 360.1, 360.2, 361.0])
    np.testing.assert_allclose(wrapped, [359.9, 0.1, 0.2, 1.0])
