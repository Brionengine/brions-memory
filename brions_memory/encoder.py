"""
Semantic + quantum encoding for Brion's Memory.

Two representations come out of every piece of content, because they do
different jobs and neither derives from the other:

    embedding      real 384-D vector -> pgvector ANN recall
    quantum_state  complex vector    -> fidelity |<s1|s2>|^2 for entanglement

The original encoder in quantum_entanglement_memory.py mapped the i-th BYTE of
the string to amplitude index i. Measured on 2026-09-12, that gave a reordered
paraphrase an overlap of 0.039 against an unrelated sentence's 0.036 -- meaning
and noise were indistinguishable, which is why realistic memories never
entangled. Both encoders below are order-invariant by construction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

STOPWORDS = frozenset("""
    a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those there their then than
""".split())


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, sort_keys=True)
    return str(content)


def tokenize(text: str) -> List[str]:
    """Content tokens plus adjacent bigrams; bigrams restore some word order."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


class Encoder:
    """
    Produces the embedding and the quantum state for a memory.

    The embedding model is loaded lazily and only once per process: on the AMD
    workers this is the expensive part of startup, and a request should never
    pay for it.
    """

    TOKEN_PROBES = 3

    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 quantum_dimension: int = EMBED_DIM,
                 allow_fallback: bool = True):
        self.model_name = model_name
        self.quantum_dimension = quantum_dimension
        self.allow_fallback = allow_fallback
        self._model = None
        self._model_failed = False

    # -- embedding ---------------------------------------------------------

    @property
    def model(self):
        if self._model is None and not self._model_failed:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded embedding model %s", self.model_name)
            except Exception as exc:
                self._model_failed = True
                if not self.allow_fallback:
                    raise
                logger.warning(
                    "Embedding model %s unavailable (%s); falling back to "
                    "token-hash encoding. Recall will not match across word "
                    "forms or synonyms until this is resolved.",
                    self.model_name, exc,
                )
        return self._model

    @property
    def using_model(self) -> bool:
        """True when real embeddings are in use. Callers should record this."""
        return self.model is not None

    def embed(self, content: Any) -> np.ndarray:
        """Real-valued semantic embedding, L2-normalised for cosine distance."""
        text = _to_text(content)
        model = self.model
        if model is not None:
            vec = np.asarray(model.encode(text), dtype=np.float32)
        else:
            vec = self._hash_embed(text)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _hash_embed(self, text: str) -> np.ndarray:
        """
        Fallback: sublinear-weighted token hashing into EMBED_DIM.

        Order-invariant and cheap. Fixes word order, does NOT fix word form --
        'dimension' and 'dimensions' remain unrelated here.
        """
        vec = np.zeros(self.quantum_dimension, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec

        freq: Dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1

        for tok, count in freq.items():
            weight = 1.0 + np.log(count)
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
            for probe in range(self.TOKEN_PROBES):
                raw = int.from_bytes(digest[probe * 4:(probe + 1) * 4], "big")
                # Signed contribution: without this, every vector sits in the
                # positive orthant and everything looks similar to everything.
                sign = 1.0 if (raw >> 31) & 1 else -1.0
                vec[raw % self.quantum_dimension] += sign * weight
        return vec

    # -- quantum state -----------------------------------------------------

    def quantum_state(self, content: Any, memory_type: str) -> np.ndarray:
        """
        Complex state for the entanglement math.

        Amplitude is |component| of the semantic embedding; the sign of each
        component becomes a phase of 0 or pi. That choice is what makes

            |<s1|s2>|^2  ==  (cosine similarity)^2

        for two memories of the same type, so the quantum fidelity inherits the
        embedding's semantic separation exactly rather than approximating it.

        An earlier version took np.abs() of the embedding and added token-hash
        phases. Measured 2026-09-12, that put unrelated text at 0.363 fidelity
        while its true cosine was 0.091 -- discarding the sign folded opposite
        meanings onto each other.

        Memory type deliberately does not enter the phase; see the note in
        the body. It is a filter, not a physical property of the content.
        """
        text = _to_text(content)
        embedding = self.embed(text).astype(np.float64)

        amplitude = np.abs(embedding)
        phase = np.where(embedding < 0, np.pi, 0.0)

        # No type term in the phase. Two earlier attempts are recorded here
        # because both measured badly and neither is worth retrying:
        #   a global type phase damped nothing (exactly 1.00000 for the same
        #   text under two types -- a global phase cancels in |<s1|s2>|^2);
        #   a type phase RAMP damped by an arbitrary amount instead, 0.892 for
        #   semantic vs episodic against 0.083 for semantic vs procedural,
        #   decided entirely by where the two type strings' hashes landed.
        # Memory type is a column with an index on it. Filter on it in the
        # query, and leave fidelity as a clean measure of meaning.

        state = amplitude * np.exp(1j * phase)
        norm = np.linalg.norm(state)
        return state / norm if norm > 0 else state

    # -- persistence helpers ----------------------------------------------

    @staticmethod
    def pack_state(state: np.ndarray) -> bytes:
        """Exact bytes for the bytea column; complex128 survives the round trip."""
        return np.asarray(state, dtype=np.complex128).tobytes()

    @staticmethod
    def unpack_state(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.complex128)

    @staticmethod
    def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """|<s1|s2>|^2 — same convention as _calculate_quantum_overlap."""
        n = min(len(state1), len(state2))
        if n == 0:
            return 0.0
        return float(abs(np.vdot(state1[:n], state2[:n])) ** 2)

    @staticmethod
    def signature(content: Any, memory_type: str) -> str:
        """Content hash used for dedupe, stable across processes."""
        payload = f"{memory_type}\x00{_to_text(content)}".encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()
