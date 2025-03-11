#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information
#  Copyright (c) 2025 Code Inc. - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Visit <https://www.codeinc.co> for more information

import enum
import gc
import logging
import os
import re
from typing import Iterable

import numpy as np
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastembed import SparseTextEmbedding, SparseEmbedding
from pydantic import BaseModel

##
# Initialize logging
##
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

##
# Config
##
MODEL_NAME = "Qdrant/bm42-all-minilm-l6-v2-attentions"
VERSION = os.getenv("VERSION") or "unknown"
BUILD_ID = os.getenv("BUILD_ID") or "unknown"
COMMIT_SHA = os.getenv("COMMIT_SHA") or "unknown"
PORT = int(os.getenv("PORT", "8000"))


##
# Models
##
class EmbedRequest(BaseModel):
    class Task(str, enum.Enum):
        QUERY: str = "query"
        INDEX: str = "index"

    class CombineStrategy(str, enum.Enum):
        MAX: str = "max"
        MEAN: str = "mean"
        SUM: str = "sum"

    texts: list[str]
    task: Task = Task.QUERY
    sparsity_threshold: float = 0.005
    allow_null_vector: bool = False
    window_size: int = 512  # Define a maximum token limit for the model
    window_overlap: int = 100  # Define a default overlap for the sliding window (in tokens)
    window_combine_strategy: CombineStrategy = CombineStrategy.MAX


class EmbedResponse(BaseModel):
    embeddings: list[tuple[list[int], list[float]] | None]
    computation_time: float = 0.0


class InfoResponse(BaseModel):
    model_name: str = MODEL_NAME
    version: str = VERSION
    build_id: str = BUILD_ID
    commit_sha: str = COMMIT_SHA


##
# Create the FastAPI app
##
app = FastAPI(
    title="Multilingual embedding API",
    description=f"API for embedding documents based on query relevance using {MODEL_NAME}",
    version=VERSION,
)

##
# Load the model
##
try:
    logger.info(f"Loading model {MODEL_NAME}...")
    model = SparseTextEmbedding(model_name=MODEL_NAME)
    logger.info(f"Model {MODEL_NAME} loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")


##
# Routes
##
@app.post("/embed", response_model=list[tuple[list[int], list[float]] | None])
def embed(request: EmbedRequest):
    """
    Embed a batch of texts into sparse vectors using sliding window approach for long texts.
    """
    texts_num = len(request.texts)
    logger.info(f"Embedding {texts_num} texts using {MODEL_NAME}")

    # Prepare result containers
    result_vectors = [None] * texts_num

    try:
        # Categorize texts based on token count
        short_texts = []  # Texts that fit within window_size
        long_texts_info = []  # Tuples of (original_index, windows) for long texts

        for idx, text in enumerate(request.texts):
            if _estimate_tokens(text) <= request.window_size:
                short_texts.append(text)
            else:
                windows = _split_into_windows(text, request.window_size, request.window_overlap)
                long_texts_info.append((idx, windows))

        # Process short texts in a batch (if any)
        if short_texts:
            # Create a mapping of original indices
            short_text_map = []
            for idx, text in enumerate(request.texts):
                if _estimate_tokens(text) <= request.window_size:
                    short_text_map.append(idx)

            # Embed all short texts and process them as a stream
            short_embeddings = _embed(short_texts, request.task)
            for embedding, orig_idx in zip(short_embeddings, short_text_map):
                sparse_vector = embedding.indices.tolist(), embedding.values.tolist()
                if request.sparsity_threshold:
                    sparse_vector = _apply_sparse_threshold(
                        sparse_vector,
                        request.sparsity_threshold,
                        request.allow_null_vector
                    )
                result_vectors[orig_idx] = sparse_vector

        # Process all windows from long texts as a single batch (if any)
        if long_texts_info:
            # Flatten all windows into a single batch
            all_windows = []
            window_map = []  # Maps each window back to its original text

            for orig_idx, windows in long_texts_info:
                for window in windows:
                    all_windows.append(window)
                    window_map.append(orig_idx)

            # Embed all windows in a single batch and process them as a stream
            all_window_embeddings = _embed(all_windows, request.task)

            # Group window embeddings by original text
            text_to_windows = {}
            for embedding, orig_idx in zip(all_window_embeddings, window_map):
                if orig_idx not in text_to_windows:
                    text_to_windows[orig_idx] = []

                sparse_vector = embedding.indices.tolist(), embedding.values.tolist()
                if request.sparsity_threshold:
                    sparse_vector = _apply_sparse_threshold(
                        sparse_vector,
                        request.sparsity_threshold,
                        request.allow_null_vector
                    )

                if sparse_vector:  # Only add non-None vectors
                    text_to_windows[orig_idx].append(sparse_vector)

            # Combine windows for each long text
            for orig_idx, window_vectors in text_to_windows.items():
                if window_vectors:
                    combined_vector = _combine_sparse_vectors(window_vectors, request.window_combine_strategy)
                    # noinspection PyTypeChecker
                    result_vectors[orig_idx] = combined_vector
                else:
                    result_vectors[orig_idx] = None if request.allow_null_vector else ([], [])
    finally:
        gc.collect()

    return result_vectors


def _embed(texts: list[str], task: EmbedRequest.Task) -> Iterable[SparseEmbedding]:
    """Embed a list of texts using the BM42 model."""
    match task:
        case EmbedRequest.Task.QUERY:
            embeddings = model.query_embed(texts)
        case EmbedRequest.Task.INDEX:
            embeddings = model.embed(texts)
        case _:
            raise ValueError(f"Unsupported task: {task}")
    return embeddings


def _split_into_windows(text: str, window_size: int, window_overlap: int) -> list[str]:
    """
    Split a text into overlapping windows based on token count.

    Args:
        text: The text to split
        window_size: Maximum size of each window in tokens
        window_overlap: Overlap between windows in tokens

    Returns:
        List of text windows
    """
    # Use a simple word-based approach with estimated token counts
    # Most tokenizers treat words as roughly 1.3 tokens on average
    tokens_per_word = 1.3
    words = re.findall(r'\w+|[^\w\s]', text)

    # Estimate the word counts based on token limits
    estimated_window_size = int(window_size / tokens_per_word)
    estimated_overlap = int(window_overlap / tokens_per_word)

    # If text fits in a single window, return it directly
    if len(words) <= estimated_window_size:
        return [text]

    # Create overlapping windows
    windows = []
    step_size = estimated_window_size - estimated_overlap

    for i in range(0, len(words), max(1, int(step_size))):
        window_words = words[i:i + estimated_window_size]
        window_text = ' '.join(window_words)
        windows.append(window_text)

    logger.info(f"Split text into {len(windows)} windows using word-based estimation")
    return windows


def _combine_sparse_vectors(
        sparse_vectors: list[tuple[list[int], list[float]]],
        strategy: EmbedRequest.CombineStrategy = EmbedRequest.CombineStrategy.MAX
) -> tuple[list[int], list[float]]:
    """
    Combine multiple sparse vectors into a single sparse vector.

    Args:
        sparse_vectors: List of sparse vectors to combine
        strategy: Strategy for combining overlapping indices
            - 'max': Take the maximum absolute value
            - 'mean': Take the average value
            - 'sum': Sum all values

    Returns:
        Combined sparse vector
    """
    if not sparse_vectors:
        return [], []

    if len(sparse_vectors) == 1:
        return sparse_vectors[0]

    # Collect all indices and their corresponding values
    all_indices = {}

    for indices, values in sparse_vectors:
        for idx, val in zip(indices, values):
            if idx not in all_indices:
                all_indices[idx] = []
            all_indices[idx].append(val)

    # Combine values for each index according to the strategy
    combined_indices = []
    combined_values = []

    for idx, vals in all_indices.items():
        combined_indices.append(idx)

        if strategy == EmbedRequest.CombineStrategy.MAX:
            # Take value with maximum absolute magnitude
            max_abs_val_idx = np.argmax(np.abs(vals))
            combined_values.append(vals[max_abs_val_idx])
        elif strategy == EmbedRequest.CombineStrategy.MEAN:
            # Take average of all values
            combined_values.append(np.mean(vals))
        elif strategy == EmbedRequest.CombineStrategy.SUM:
            # Sum all values
            combined_values.append(np.sum(vals))
        else:
            # Default to max
            max_abs_val_idx = np.argmax(np.abs(vals))
            combined_values.append(vals[max_abs_val_idx])

    # Ensure indices are sorted (optional, but can be helpful)
    sorted_idx = np.argsort(combined_indices)
    combined_indices = [combined_indices[i] for i in sorted_idx]
    combined_values = [combined_values[i] for i in sorted_idx]

    return combined_indices, combined_values


def _apply_sparse_threshold(
        sparse_vector: tuple[list[int], list[float]],
        sparsity_threshold: float,
        allow_null_vector: bool
) -> tuple[list[int], list[float]] | None:
    """
    Filter out values below the sparsity threshold.

    Args:
        sparse_vector: A sparse vector as (indices, values) pair
        sparsity_threshold: The threshold for sparsity
        allow_null_vector: Whether to allow null vectors

    Returns:
        A filtered sparse vector
    """
    indices, values = sparse_vector

    # Check if values is empty before proceeding
    if not values:
        return ([], []) if not allow_null_vector else None

    filtered_indices: list[int] = []
    filtered_values: list[float] = []
    for i, value in enumerate(values):
        if abs(value) >= sparsity_threshold:
            filtered_indices.append(indices[i])
            filtered_values.append(value)

    # If all values were filtered out, keep the highest magnitude value
    if not filtered_values:
        if allow_null_vector:
            return None
        else:
            # Only try to find max if values is not empty
            if values:
                max_idx = np.argmax(np.abs(values))
                filtered_indices = [indices[max_idx]]
                filtered_values = [values[max_idx]]
            else:
                # Return empty lists if values is empty
                return [], []

    return filtered_indices, filtered_values


def _estimate_tokens(text: str) -> int:
    """
    Count the number of tokens in a text.
    """
    return int(len(re.findall(r'\w+|[^\w\s]', text)) * 1.3)


@app.get("/info", response_model=InfoResponse)
def info():
    """
    Get information about the model and the API.
    """
    return InfoResponse()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
