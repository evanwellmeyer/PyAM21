import os
import pytest
import torch

from pyam21.testing import DUMPS

torch.set_default_dtype(torch.float64)


def have_dumps(tag):
    return os.path.isdir(os.path.join(DUMPS, tag)) and len(os.listdir(os.path.join(DUMPS, tag))) > 0


def requires_dumps(tag):
    if not have_dumps(tag):
        pytest.skip(f"reference dumps '{tag}' not present (run reference/scripts/make_dumps.sh)")
