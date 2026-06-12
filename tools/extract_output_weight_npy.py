#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deprecated standalone extractor.

Prefer the Megatron-integrated path (loads checkpoint with the same args as training):

  ./experiments/extract_output_weight_qwen.sh

That runs ``run_job.sh`` with ``--extract-output-weight-npy=/path/to/output_weight.npy``.
"""

from __future__ import annotations

import sys

print(__doc__, file=sys.stderr)
raise SystemExit(
    "Use experiments/extract_output_weight_qwen.sh instead of this script."
)
