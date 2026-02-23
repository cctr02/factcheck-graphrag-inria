"""
Checkpoint manager for the knowledge graph build pipeline.

Saves and loads intermediate results at key pipeline stages,
enabling resume after interruption (e.g., HDBSCAN crash/hang).

Checkpoint stages (ordered):
    step_01_chunks       - After loading/chunking documents + claims
    step_02a_embeddings  - After computing chunk embeddings (GPU)
    step_02b_topics      - After BERTopic fit_transform (UMAP + HDBSCAN + c-TF-IDF)
    step_03_graph_themes - After adding themes + hierarchical topics
    step_04_graph_entities - After entity extraction + keyphrases
    step_05_graph_edges  - After all cross-edges (CO_OCCURS, SEMANTIC_SIM, etc.)
    step_06_graph_claims - After claims + pre-retrieval

Usage:
    # Fresh build
    cm = CheckpointManager(Path("outputs/checkpoints"), logger=logger)
    cm.clear()
    cm.init_run(config, max_docs)
    cm.save("step_01_chunks", {"chunks": chunks, ...})

    # Resume
    cm = CheckpointManager(Path("outputs/checkpoints"), logger=logger)
    if cm.should_skip("step_01_chunks"):
        data = cm.load("step_01_chunks")
"""

import json
import pickle
import hashlib
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any


# Ordered list of checkpoint steps
CHECKPOINT_STEPS = [
    "step_01_chunks",
    "step_02a_embeddings",
    "step_02b_topics",
    "step_03_graph_themes",
    "step_04_graph_entities",
    "step_05_graph_edges",
    "step_06_graph_claims",
]


class CheckpointManager:
    """
    Manages checkpoint save/load for the build pipeline.

    Checkpoints are stored in a directory with:
    - state.json: Tracks last completed step + config hash
    - step_*.pkl / step_*.npz: Serialized intermediate data
    """

    def __init__(self, checkpoint_dir: Path, logger: Optional[logging.Logger] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.state_path = self.checkpoint_dir / "state.json"
        self._state = None

    def _load_state(self) -> dict:
        """Load state from disk, or return empty state."""
        if self._state is not None:
            return self._state

        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                self._state = json.load(f)
        else:
            self._state = {}

        return self._state

    def _save_state(self, state: dict):
        """Save state to disk atomically."""
        self._state = state
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename (atomic on most OS)
        temp_path = self.state_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(state, f, indent=2)
        temp_path.replace(self.state_path)

    def init_run(self, config, max_docs: int):
        """
        Initialize a new checkpoint run. Saves config hash for validation.

        Args:
            config: Build configuration object
            max_docs: Max documents (from CLI or config)
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        config_hash = self._compute_config_hash(config, max_docs)

        state = {
            "created_at": datetime.now().isoformat(),
            "config_hash": config_hash,
            "max_docs": max_docs,
            "last_completed": None,
            "steps": {}
        }
        self._save_state(state)
        self.logger.info(f"[CHECKPOINT] New run initialized in {self.checkpoint_dir}")

    def _compute_config_hash(self, config, max_docs: int) -> str:
        """
        Compute hash of critical config parameters.
        If these change, checkpoints are invalid.
        """
        critical_params = {
            "max_docs": max_docs,
            "chunk_size": config.graph_construction.chunk_size,
            "chunk_overlap": config.graph_construction.chunk_overlap,
            "embedding_model": config.models.embedding_model,
            "ner_model": config.models.ner_model,
            "min_topic_size": config.graph_construction.min_topic_size,
            "nr_topics": config.graph_construction.nr_topics,
            "entity_types": sorted(config.graph_construction.entity_types),
        }
        params_json = json.dumps(critical_params, sort_keys=True)
        return hashlib.sha256(params_json.encode()).hexdigest()[:16]

    def validate_config(self, config, max_docs: int) -> bool:
        """
        Check if current config matches the checkpoint's config.

        Returns:
            True if config matches, False if it changed.
        """
        state = self._load_state()
        if not state:
            return True  # No state = no conflict

        saved_hash = state.get("config_hash", "")
        current_hash = self._compute_config_hash(config, max_docs)

        if saved_hash != current_hash:
            self.logger.error(f"[CHECKPOINT] Config hash mismatch!")
            self.logger.error(f"  Checkpoint: {saved_hash}")
            self.logger.error(f"  Current:    {current_hash}")
            return False

        return True

    def get_last_completed(self) -> Optional[str]:
        """
        Get the name of the last completed checkpoint step.

        Returns:
            Step name (e.g., "step_02a_embeddings") or None if no checkpoints exist.
        """
        state = self._load_state()
        return state.get("last_completed")

    def should_skip(self, step_name: str) -> bool:
        """
        Check if a step should be skipped (already checkpointed).

        Args:
            step_name: Step to check (e.g., "step_02a_embeddings")

        Returns:
            True if the step was already completed and checkpoint file exists.
        """
        state = self._load_state()
        last_completed = state.get("last_completed")

        if not last_completed:
            return False

        # Check if step_name is before or equal to last_completed in the ordered list
        if step_name not in CHECKPOINT_STEPS or last_completed not in CHECKPOINT_STEPS:
            return False

        step_idx = CHECKPOINT_STEPS.index(step_name)
        last_idx = CHECKPOINT_STEPS.index(last_completed)

        if step_idx > last_idx:
            return False  # This step hasn't been completed yet

        # Verify the checkpoint file actually exists
        checkpoint_path = self._get_checkpoint_path(step_name)
        if not checkpoint_path.exists():
            self.logger.warning(f"[CHECKPOINT] Step {step_name} marked as done but file missing!")
            return False

        return True

    def _get_checkpoint_path(self, step_name: str) -> Path:
        """Get the file path for a checkpoint."""
        if step_name == "step_02a_embeddings":
            return self.checkpoint_dir / f"{step_name}.npz"
        else:
            return self.checkpoint_dir / f"{step_name}.pkl"

    def save(self, step_name: str, data: dict, metadata: Optional[dict] = None):
        """
        Save a checkpoint.

        Args:
            step_name: Step name (e.g., "step_02a_embeddings")
            data: Dictionary of data to save
            metadata: Optional metadata to store in state.json (small info only)
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self._get_checkpoint_path(step_name)

        start_time = datetime.now()

        # Save data based on format
        if step_name == "step_02a_embeddings":
            # Use compressed numpy for embeddings (more efficient)
            np.savez_compressed(checkpoint_path, **{
                k: v for k, v in data.items()
                if isinstance(v, np.ndarray)
            })
            # Save non-numpy data separately if any
            non_numpy = {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}
            if non_numpy:
                meta_path = self.checkpoint_dir / f"{step_name}_meta.pkl"
                with open(meta_path, 'wb') as f:
                    pickle.dump(non_numpy, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Use pickle for everything else
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.replace(checkpoint_path)

        elapsed = (datetime.now() - start_time).total_seconds()

        # Update state
        state = self._load_state()
        state["last_completed"] = step_name
        state["updated_at"] = datetime.now().isoformat()

        step_info = {
            "completed_at": datetime.now().isoformat(),
            "save_time_seconds": round(elapsed, 2),
            "file_size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 1)
        }
        if metadata:
            step_info.update(metadata)

        state.setdefault("steps", {})[step_name] = step_info
        self._save_state(state)

        size_mb = step_info["file_size_mb"]
        self.logger.info(f"[CHECKPOINT] Saved: {step_name} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    def load(self, step_name: str) -> dict:
        """
        Load a checkpoint.

        Args:
            step_name: Step name to load

        Returns:
            Dictionary of saved data

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = self._get_checkpoint_path(step_name)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        start_time = datetime.now()

        if step_name == "step_02a_embeddings":
            # Load numpy data
            npz_data = np.load(checkpoint_path)
            data = {key: npz_data[key] for key in npz_data.files}
            # Load non-numpy metadata if exists
            meta_path = self.checkpoint_dir / f"{step_name}_meta.pkl"
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                data.update(meta)
        else:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)

        elapsed = (datetime.now() - start_time).total_seconds()
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"[CHECKPOINT] Loaded: {step_name} ({size_mb:.1f} MB, {elapsed:.1f}s)")

        return data

    def clear(self):
        """Delete all checkpoint files."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            self.logger.info(f"[CHECKPOINT] Cleared all checkpoints in {self.checkpoint_dir}")
        self._state = None

    def get_summary(self) -> dict:
        """Get a summary of current checkpoint state for display."""
        state = self._load_state()
        if not state:
            return {"status": "no_checkpoints"}

        steps = state.get("steps", {})
        total_size = 0
        for step_name in steps:
            path = self._get_checkpoint_path(step_name)
            if path.exists():
                total_size += path.stat().st_size

        return {
            "status": "has_checkpoints",
            "created_at": state.get("created_at", "unknown"),
            "last_completed": state.get("last_completed"),
            "max_docs": state.get("max_docs"),
            "completed_steps": list(steps.keys()),
            "total_size_mb": round(total_size / (1024 * 1024), 1)
        }
