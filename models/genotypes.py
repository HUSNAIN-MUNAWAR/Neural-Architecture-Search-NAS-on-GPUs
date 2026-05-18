"""Predefined DARTS genotypes for evaluation and smoke tests."""

from __future__ import annotations

from typing import Dict, List


DARTS_V1: Dict[str, List] = {
    "normal": [
        ["sep_conv_3x3", 1],
        ["sep_conv_3x3", 0],
        ["skip_connect", 0],
        ["sep_conv_3x3", 1],
        ["skip_connect", 0],
        ["sep_conv_3x3", 1],
        ["sep_conv_3x3", 0],
        ["skip_connect", 2],
    ],
    "normal_concat": [2, 3, 4, 5],
    "reduce": [
        ["max_pool_3x3", 0],
        ["max_pool_3x3", 1],
        ["skip_connect", 2],
        ["max_pool_3x3", 0],
        ["max_pool_3x3", 0],
        ["skip_connect", 2],
        ["skip_connect", 2],
        ["avg_pool_3x3", 0],
    ],
    "reduce_concat": [2, 3, 4, 5],
}


DARTS_V2: Dict[str, List] = {
    "normal": [
        ["sep_conv_3x3", 0],
        ["sep_conv_3x3", 1],
        ["sep_conv_3x3", 0],
        ["sep_conv_3x3", 1],
        ["sep_conv_3x3", 1],
        ["skip_connect", 0],
        ["skip_connect", 0],
        ["dil_conv_3x3", 2],
    ],
    "normal_concat": [2, 3, 4, 5],
    "reduce": [
        ["max_pool_3x3", 0],
        ["max_pool_3x3", 1],
        ["skip_connect", 2],
        ["max_pool_3x3", 1],
        ["max_pool_3x3", 0],
        ["skip_connect", 2],
        ["skip_connect", 2],
        ["max_pool_3x3", 1],
    ],
    "reduce_concat": [2, 3, 4, 5],
}
