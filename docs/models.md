# Verified Model Hashes

CANVAS uses [sentence-transformers](https://www.sbert.net/) models for intent embedding.
This page documents SHA-256 hashes of the primary weights files, enabling integrity
verification and air-gapped (offline) deployment.

## Supported Models

| Model | Default? | Weights File | SHA-256 |
|-------|----------|-------------|---------|
| `all-MiniLM-L6-v2` | Yes | `model.safetensors` | *(see below)* |
| `paraphrase-multilingual-MiniLM-L12-v2` | No | `model.safetensors` | *(see below)* |

> **Note**: Hash entries in `KNOWN_MODEL_HASHES` are empty by default to avoid breaking
> installs where the model has not yet been downloaded. Populate them after your first run
> (see _Computing Hashes_ below) and pin the value for reproducible enterprise deployments.

## Computing Hashes

After downloading a model, run:

```python
from canvas_heal.embedder import compute_model_hash
print(compute_model_hash("all-MiniLM-L6-v2"))
# e.g. a1b2c3d4e5f6...
```

Then update `KNOWN_MODEL_HASHES` in `canvas_heal/embedder.py`, or pass the hash explicitly:

```python
from canvas_heal.embedder import IntentEmbedder
embedder = IntentEmbedder.get(
    "all-MiniLM-L6-v2",
    model_sha256="a1b2c3d4e5f6...",   # your verified hash
)
```

A `ModelIntegrityError` is raised immediately if the hash does not match.

## Air-Gapped Deployment

Set `CANVAS_MODEL_CACHE_DIR` to a directory containing pre-downloaded models before
starting your process:

```bash
export CANVAS_MODEL_CACHE_DIR=/opt/canvas-models
```

Then pre-bundle the model into that directory on an internet-connected machine:

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/opt/canvas-models")
```

Copy `/opt/canvas-models` to the air-gapped host. CANVAS will load from that path
without making any network calls. Combine with a pinned `model_sha256` to ensure
the bundled weights have not been tampered with.

## Updating Hashes After a Model Release

HuggingFace periodically updates model weights. When upgrading:

1. Download the new weights on an internet-connected machine.
2. Run `compute_model_hash` to get the new SHA-256.
3. Update `KNOWN_MODEL_HASHES` and this document.
4. Commit the change so all downstream consumers receive the new expected hash.
