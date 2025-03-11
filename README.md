# Multilingual Embedding API

A FastAPI service for generating sparse text embeddings optimized for query relevance using the
[`Qdrant/bm42-all-minilm-l6-v2-attentions`](https://qdrant.tech/articles/bm42/) model.

## Overview

This API provides endpoints for converting text into sparse vector representations (embeddings) that can be used for
semantic search, document retrieval, and other NLP applications. It's particularly designed to handle long documents
through a sliding window approach.

## Features

- Generate query-optimized embeddings for search
- Generate index-optimized embeddings for document storage
- Handle long texts with configurable sliding window approach
- Apply sparsity thresholds to optimize vector storage
- Multiple strategies for combining window embeddings (max, mean, sum)

## API Endpoints

### `/embed` (POST)

Converts a batch of texts into sparse embeddings.

**Request Body:**

```json
{
  "texts": [
    "your text here",
    "another text"
  ],
  "task": "query",
  "sparsity_threshold": 0.005,
  "allow_null_vector": false,
  "window_size": 512,
  "window_overlap": 100,
  "window_combine_strategy": "max"
}
```

**Parameters:**

- `texts`: List of texts to embed
- `task`: Either "query" (for search queries) or "index" (for documents to be stored)
- `sparsity_threshold`: Filter out vector dimensions with values below this threshold
- `allow_null_vector`: Whether to allow null vectors when all values are below threshold
- `window_size`: Maximum token limit for each processing window
- `window_overlap`: Overlap between adjacent windows in tokens
- `window_combine_strategy`: Strategy for combining window vectors ("max", "mean", "sum")

**Response:**
Returns a list of sparse vectors in the format `[indices, values]` or `null` (if vector is filtered out).

### `/info` (GET)

Returns information about the API, including:

- Model name
- API version
- Build ID
- Commit SHA

## Installation

### Requirements

- Python 3.10+
- FastAPI
- fastembed
- numpy
- pydantic

### Environment Variables

- `VERSION`: API version
- `BUILD_ID`: Build identifier
- `COMMIT_SHA`: Git commit SHA
- `PORT`: Server port (default: 8000)

## Usage

### Starting the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker run -p 8000:8000 joanfabregat/bm42-embed:latest
```

## Implementation Details

### Sliding Window Approach

For long texts exceeding the token limit, the API splits the text into overlapping windows, processes each window
separately, and then combines the resulting embeddings using the specified strategy.

### Sparsity Handling

The API can filter out dimensions with small values to create sparser vectors, which can significantly reduce storage
requirements while maintaining semantic quality.

## License

This project is licensed under the MIT License - see the license notice in the code for details.

## Author

Developed by Joan Fabrégat, j@fabreg.at