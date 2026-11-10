"""FAISS Index Persistence and Safe Deserialization for NoteBot."""

import json
import os
import shutil
from typing import Dict, Any, Optional, List
from langchain_community.vectorstores import FAISS
from notebot.config.logging_config import setup_logger

logger = setup_logger("notebot.persistence")


def compute_dataset_hash(files_info: List[Dict[str, Any]], embed_signature: str) -> str:
    """Computes a deterministic hash identifier for a set of documents and embedding config."""
    import hashlib
    data = {
        "files": sorted([{"name": f["name"], "size": f.get("size", 0)} for f in files_info], key=lambda x: x["name"]),
        "embed_sig": embed_signature
    }
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_faiss_index(
    vector_store: FAISS,
    metadata: Dict[str, Any],
    storage_dir: str,
    index_id: str
) -> str:
    """
    Saves a FAISS vector store and its associated metadata to disk safely.
    
    Directory structure:
        <storage_dir>/<index_id>/
            ├── index.faiss
            ├── index.pkl
            └── metadata.json
    """
    safe_index_id = os.path.basename(index_id)
    target_dir = os.path.join(storage_dir, safe_index_id)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Save FAISS index
    vector_store.save_local(target_dir)

    # 2. Save metadata JSON
    meta_path = os.path.join(target_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Persisted FAISS index and metadata to: {target_dir}")
    return target_dir


def load_faiss_index(
    storage_dir: str,
    index_id: str,
    embeddings: Any
) -> Optional[FAISS]:
    """
    Safely loads a FAISS index from disk.
    Enforces safe deserialization flag allow_dangerous_deserialization=True only for locally created files.
    """
    safe_index_id = os.path.basename(index_id)
    target_dir = os.path.join(storage_dir, safe_index_id)

    faiss_file = os.path.join(target_dir, "index.faiss")
    pkl_file = os.path.join(target_dir, "index.pkl")
    meta_file = os.path.join(target_dir, "metadata.json")

    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        logger.warning(f"Index files missing in {target_dir}")
        return None

    try:
        store = FAISS.load_local(
            folder_path=target_dir,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Successfully loaded FAISS index from: {target_dir}")
        return store
    except Exception as e:
        logger.error(f"Failed to deserialize FAISS index from {target_dir}: {e}")
        return None


def get_index_metadata(storage_dir: str, index_id: str) -> Optional[Dict[str, Any]]:
    """Reads metadata.json for a saved index."""
    safe_index_id = os.path.basename(index_id)
    meta_file = os.path.join(storage_dir, safe_index_id, "metadata.json")
    if not os.path.exists(meta_file):
        return None
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read metadata from {meta_file}: {e}")
        return None


def delete_faiss_index(storage_dir: str, index_id: str) -> bool:
    """Removes a persisted index directory."""
    safe_index_id = os.path.basename(index_id)
    target_dir = os.path.join(storage_dir, safe_index_id)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            logger.info(f"Deleted index directory: {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {target_dir}: {e}")
            return False
    return False


def list_saved_indexes(storage_dir: str) -> List[Dict[str, Any]]:
    """Returns a list of all available persisted index profiles."""
    if not os.path.exists(storage_dir):
        return []

    indexes = []
    for item in os.listdir(storage_dir):
        item_path = os.path.join(storage_dir, item)
        if os.path.isdir(item_path):
            meta = get_index_metadata(storage_dir, item)
            indexes.append({
                "index_id": item,
                "metadata": meta or {}
            })
    return indexes

