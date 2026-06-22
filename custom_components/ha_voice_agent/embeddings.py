"""Entity embedding index for semantic search — Phase 6C."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import httpx

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import area_registry as ar, entity_registry as er

_LOGGER = logging.getLogger(__name__)

# Rebuild debounce — wait this many seconds after a registry change before rebuilding
_REBUILD_DEBOUNCE = 5.0


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (fallback if numpy unavailable)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if (norm_a and norm_b) else 0.0


def _batch_cosine_sim(query: list[float], matrix: list[list[float]]) -> list[float]:
    """Compute cosine similarity between query and each row in matrix."""
    try:
        import numpy as np  # HA has numpy; this is fast for 182 × 768

        q = np.array(query, dtype=np.float32)
        m = np.array(matrix, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        m_norms = np.linalg.norm(m, axis=1)
        denom = q_norm * m_norms
        denom[denom == 0] = 1e-9
        return (m @ q / denom).tolist()
    except ImportError:
        return [_cosine_sim(query, row) for row in matrix]


class EntityEmbeddingIndex:
    """
    Maintains an in-memory vector index of all exposed entities.

    On each user query, embeds the query text and returns the top-K most
    similar entity_ids so the LLM prompt stays small (~15-25 entities
    instead of all 182).
    """

    def __init__(
        self,
        hass: HomeAssistant,
        ollama_url: str,
        embed_model: str,
        top_k: int = 15,
    ) -> None:
        self.hass = hass
        self.ollama_url = ollama_url.rstrip("/")
        self.embed_model = embed_model
        self.top_k = top_k

        self._entity_ids: list[str] = []
        self._embeddings: list[list[float]] = []
        self._ready = False
        self._building = False
        self._rebuild_handle: asyncio.TimerHandle | None = None
        self._unsub: list[Any] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def async_setup(self, entities: list[dict]) -> None:
        """Initial index build and registry-change listeners."""
        await self._build(entities)
        self._unsub.append(
            self.hass.bus.async_listen(
                er.EVENT_ENTITY_REGISTRY_UPDATED, self._on_registry_change
            )
        )
        self._unsub.append(
            self.hass.bus.async_listen(
                ar.EVENT_AREA_REGISTRY_UPDATED, self._on_registry_change
            )
        )

    def async_teardown(self) -> None:
        """Unsubscribe listeners."""
        for unsub in self._unsub:
            unsub()
        self._unsub.clear()

    async def async_search(self, query: str) -> list[str]:
        """
        Return up to top_k entity_ids most semantically similar to query.
        Falls back to returning all entity_ids if the index isn't ready.
        """
        if not self._ready or not self._embeddings:
            _LOGGER.debug("Embedding index not ready, returning all entities")
            return self._entity_ids

        try:
            query_vec = await self._embed([query])
            if not query_vec:
                return self._entity_ids

            scores = _batch_cosine_sim(query_vec[0], self._embeddings)
            ranked = sorted(
                zip(scores, self._entity_ids), key=lambda x: x[0], reverse=True
            )
            top = [eid for _, eid in ranked[: self.top_k]]
            _LOGGER.debug(
                "Vector search top-%d for %r → %s", self.top_k, query[:60], top
            )
            return top
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Vector search failed, falling back to all: %s", err)
            return self._entity_ids

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @callback
    def _on_registry_change(self, event: Event) -> None:
        """Debounced rebuild when the entity or area registry changes."""
        if self._rebuild_handle:
            self._rebuild_handle.cancel()
        self._rebuild_handle = self.hass.loop.call_later(
            _REBUILD_DEBOUNCE, self._schedule_rebuild
        )

    @callback
    def _schedule_rebuild(self) -> None:
        self.hass.async_create_task(self._rebuild())

    async def _rebuild(self) -> None:
        from homeassistant.helpers.entity_registry import async_get as er_get

        entity_reg = er_get(self.hass)
        entities = []
        for state in self.hass.states.async_all():
            if not async_should_expose(
                self.hass, conversation.DOMAIN, state.entity_id
            ):
                continue
            e = entity_reg.async_get(state.entity_id)
            entities.append(
                {
                    "entity_id": state.entity_id,
                    "name": state.name,
                    "aliases": (e.aliases or []) if e else [],
                }
            )
        await self._build(entities)

    async def _build(self, entities: list[dict]) -> None:
        """Batch-embed all entity descriptions and store the index."""
        if self._building or not entities:
            return
        self._building = True
        try:
            area_reg = ar.async_get(self.hass)
            entity_reg = er.async_get(self.hass)

            descriptions: list[str] = []
            entity_ids: list[str] = []

            for entity in entities:
                eid = entity["entity_id"]
                name = entity["name"]
                domain = eid.split(".")[0]
                aliases = entity.get("aliases") or []

                e = entity_reg.async_get(eid)
                area_name = ""
                if e and e.area_id:
                    area = area_reg.async_get_area(e.area_id)
                    if area:
                        area_name = area.name

                # Rich description: name + area + domain + aliases
                parts = [name]
                if area_name:
                    parts.append(f"in {area_name}")
                parts.append(f"({domain})")
                if aliases:
                    parts.append("aka " + ", ".join(aliases[:2]))

                descriptions.append(" ".join(parts))
                entity_ids.append(eid)

            _LOGGER.info(
                "Building embedding index for %d entities using %s",
                len(descriptions),
                self.embed_model,
            )

            embeddings = await self._embed(descriptions)
            if not embeddings or len(embeddings) != len(descriptions):
                _LOGGER.error(
                    "Embedding mismatch: got %d for %d entities",
                    len(embeddings) if embeddings else 0,
                    len(descriptions),
                )
                return

            self._entity_ids = entity_ids
            self._embeddings = embeddings
            self._ready = True
            _LOGGER.info("Embedding index ready: %d entities", len(entity_ids))

        except Exception as err:  # noqa: BLE001
            _LOGGER.error("Failed to build embedding index: %s", err)
        finally:
            self._building = False

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed and return list of embedding vectors."""
        url = f"{self.ollama_url}/api/embed"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={"model": self.embed_model, "input": texts},
                timeout=60.0,
            )
            resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])
