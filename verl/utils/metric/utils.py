# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics utils.
"""

from typing import Any

import numpy as np


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics

def compute_token_ttr(sequence: list[int], n: int = 1) -> float:
    if len(sequence) < n:
        return 0.0

    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i : i + n])
        ngrams.append(ngram)

    if len(ngrams) > 0:
        ttr = len(set(ngrams)) / len(ngrams)
        return ttr

    return 0.0

def compute_text_ttr(text: str, n: int = 1) -> float:
    words = text.split()

    if len(words) < n:
        return 0.0

    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i : i + n])
        ngrams.append(ngram)

    if len(ngrams) > 0:
        ttr = len(set(ngrams)) / len(ngrams)
        return ttr

    return 0.0
import torch
def compute_token_ttr_batch(sequences: torch.Tensor, n: int = 1) -> torch.Tensor:
    """
    Compute token type-to-token ratio (TTR) for batched sequences.
    
    Args:
        sequences: [bsz, seq_length] tensor of token ids
        n: n-gram size (default: 1 for unigrams)
    
    Returns:
        [bsz, 1] tensor of TTR values
    """
    bsz, seq_length = sequences.shape
    
    if seq_length < n:
        return torch.zeros(bsz, 1, device=sequences.device)
    
    ttrs = []
    
    for i in range(bsz):
        seq = sequences[i].tolist()  # Convert to list for easier processing
        
        # Extract n-grams
        ngrams = []
        for j in range(len(seq) - n + 1):
            ngram = tuple(seq[j : j + n])
            ngrams.append(ngram)
        
        if len(ngrams) > 0:
            ttr = len(set(ngrams)) / len(ngrams)
        else:
            ttr = 0.0
        
        ttrs.append(ttr)
    
    return torch.tensor(ttrs, device=sequences.device).unsqueeze(1)