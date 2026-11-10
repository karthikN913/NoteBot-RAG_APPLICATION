"""Unit Tests for FAISS Index Persistence and Deserialization."""

import os
import shutil
import pytest
from notebot.vectorstore.persistence import (
    save_faiss_index,
    load_faiss_index,
    get_index_metadata,
    delete_faiss_index,
    list_saved_indexes,
    compute_dataset_hash
)
from notebot.vectorstore.faiss_manager import FAISSIndexManager


def test_index_persistence_lifecycle(sample_documents, mock_embeddings, tmp_path):
    storage_dir = str(tmp_path / "indexes")
    manager = FAISSIndexManager.from_documents(sample_documents, mock_embeddings)

    metadata = {
        "index_id": "test_idx_1",
        "documents": ["AI_Intro.pdf", "Deep_Learning.pdf"],
        "chunk_count": len(sample_documents),
        "embedding_provider": "mock"
    }

    # 1. Save index
    saved_path = save_faiss_index(
        vector_store=manager.vector_store,
        metadata=metadata,
        storage_dir=storage_dir,
        index_id="test_idx_1"
    )
    assert os.path.exists(saved_path)
    assert os.path.exists(os.path.join(saved_path, "index.faiss"))
    assert os.path.exists(os.path.join(saved_path, "metadata.json"))

    # 2. Check metadata
    read_meta = get_index_metadata(storage_dir, "test_idx_1")
    assert read_meta is not None
    assert read_meta["chunk_count"] == 4

    # 3. List indexes
    idx_list = list_saved_indexes(storage_dir)
    assert len(idx_list) == 1
    assert idx_list[0]["index_id"] == "test_idx_1"

    # 4. Load index
    loaded_store = load_faiss_index(storage_dir, "test_idx_1", mock_embeddings)
    assert loaded_store is not None
    results = loaded_store.similarity_search("neural networks", k=1)
    assert len(results) == 1

    # 5. Delete index
    deleted = delete_faiss_index(storage_dir, "test_idx_1")
    assert deleted is True
    assert not os.path.exists(saved_path)


def test_compute_dataset_hash():
    files_a = [{"name": "doc1.pdf", "size": 1000}, {"name": "doc2.pdf", "size": 2000}]
    files_b = [{"name": "doc2.pdf", "size": 2000}, {"name": "doc1.pdf", "size": 1000}]
    # Order independence
    assert compute_dataset_hash(files_a, "google_gemini-embedding-001") == compute_dataset_hash(files_b, "google_gemini-embedding-001")

