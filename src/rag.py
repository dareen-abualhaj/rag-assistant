"""RAG pipeline: retrieve -> relevance gate -> generate -> verify -> log latency."""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple

from src.embeddings import load_model
from src.llm import LLMConfigError, LLMError, generate
from src.prompts import NO_ANSWER, build_context, build_messages
from src.vector_store import get_collection, search

logger = logging.getLogger(__name__)

LOG_PATH = "logs/latency.jsonl"
MAX_DISTANCE = 0.45          # cosine distance; in-domain top hits were 0.23-0.36. Calibrate!
GROUNDING_THRESHOLD = 0.30  # min cosine similarity between an answer sentence and the sources
CITATION_RE = re.compile(r"\[(\d+)\]")


class RAGPipeline:
    def __init__(self, embedder=None, collection=None, llm: Callable = generate,
                 k: int = 5, max_distance: float = MAX_DISTANCE, log_path: str = LOG_PATH):
        self.embedder = embedder if embedder is not None else load_model()
        self.collection = collection if collection is not None else get_collection()
        self.llm = llm
        self.k = k
        self.max_distance = max_distance
        self.log_path = log_path

    # ---------- helpers ----------
    def _embed(self, texts: List[str]):
        return self.embedder.encode(texts, normalize_embeddings=True)

    @staticmethod
    def check_citations(answer: str, n_sources: int) -> bool:
        """True if the answer cites at least one source and every citation is valid."""
        cited = {int(m) for m in CITATION_RE.findall(answer)}
        return bool(cited) and all(1 <= c <= n_sources for c in cited)

    def check_grounding(self, answer: str, used: List[Dict]) -> Tuple[float, List[str]]:
        """Embedding-based check: is every answer sentence similar to some source chunk?"""
        clean = CITATION_RE.sub("", answer)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if len(s.strip()) >= 20]
        if not sentences:
            return 1.0, []
        chunk_emb = self._embed([h["text"] for h in used])
        sent_emb = self._embed(sentences)
        best = (sent_emb @ chunk_emb.T).max(axis=1)
        ungrounded = [s for s, b in zip(sentences, best) if b < GROUNDING_THRESHOLD]
        return float(best.mean()), ungrounded

    def _verify(self, text: str, used: List[Dict]) -> Tuple[List[str], float]:
        """Run both hallucination checks. Returns (warnings, grounding_score)."""
        if NO_ANSWER.lower() in text.lower():
            return [], 1.0
        warnings = []
        if not self.check_citations(text, len(used)):
            warnings.append("missing or invalid source citations")
        score, ungrounded = self.check_grounding(text, used)
        if ungrounded:
            warnings.append(f"{len(ungrounded)} sentence(s) not supported by sources")
        return warnings, score

    def _log(self, result: Dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": result["question"],
            "status": result["status"],
            "top_distance": result["top_distance"],
            "n_sources": len(result["sources"]),
            **result["latency"],
        }
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.warning("Could not write latency log: %s", e)

    # ---------- main entry ----------
    def answer(self, question: str) -> Dict:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty")

        t0 = time.perf_counter()
        q_emb = self._embed([question])[0]
        hits = search(self.collection, q_emb, k=self.k)
        t1 = time.perf_counter()

        result = {
            "question": question,
            "answer": "",
            "status": "",
            "sources": [],
            "warnings": [],
            "top_distance": round(hits[0]["distance"], 4) if hits else None,
        }

        # Gate 1: off-topic / out-of-domain -> do not even call the LLM
        if not hits or hits[0]["distance"] > self.max_distance:
            result["status"] = "out_of_domain"
            result["answer"] = NO_ANSWER
            t2 = t3 = t1
        else:
            context, used = build_context(hits)
            result["sources"] = [
                {"n": i + 1, "title": h["metadata"].get("title"),
                 "url": h["metadata"].get("url"), "distance": round(h["distance"], 4)}
                for i, h in enumerate(used)
            ]
            try:
                text = self.llm(build_messages(question, context))
            except LLMError as e:
                logger.error("LLM failure: %s", e)
                text = None
                result["status"] = "error"
                if isinstance(e, LLMConfigError):
                    result["answer"] = "The answer service is misconfigured. Please contact the administrator."
                else:
                    result["answer"] = "Sorry, the answer service is temporarily unavailable. Please try again."
                result["error"] = str(e)
            t2 = time.perf_counter()

            # Gate 2: verify; on failure retry once with a stricter prompt, else refuse
            if text is not None:
                warnings, score = self._verify(text, used)
                if warnings:
                    logger.warning("Verification failed (%s); retrying with strict prompt", warnings)
                    try:
                        text2 = self.llm(build_messages(question, context, strict=True))
                        warnings2, score2 = self._verify(text2, used)
                        if not warnings2:
                            text, warnings, score = text2, warnings2, score2
                    except LLMError as e:
                        logger.error("LLM retry failure: %s", e)
                if NO_ANSWER.lower() in text.lower():
                    result["status"] = "no_answer"
                    result["answer"] = NO_ANSWER
                elif warnings:
                    result["status"] = "unverified"
                    result["answer"] = ("I can't answer this reliably from the knowledge base. "
                                        "Please check the sources below.")
                    result["rejected_answer"] = text  # debugging only
                    result["warnings"] = warnings
                else:
                    result["status"] = "answered"
                    result["answer"] = text
                result["grounding_score"] = round(score, 3)
            t3 = time.perf_counter()

        result["latency"] = {
            "retrieval_ms": round((t1 - t0) * 1000, 1),
            "generation_ms": round((t2 - t1) * 1000, 1),
            "verification_ms": round((t3 - t2) * 1000, 1),
            "total_ms": round((t3 - t0) * 1000, 1),
        }
        logger.info("status=%s total=%.0fms", result["status"], result["latency"]["total_ms"])
        self._log(result)
        return result