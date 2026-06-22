"""HA Voice Agent — conversation entity."""

from __future__ import annotations

import logging
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
    async_get_chat_log,
)
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
    template,
)
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_EMBED_MODEL,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_CALLS,
    CONF_MODEL,
    CONF_NUM_CTX,
    CONF_OLLAMA_URL,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_K,
    CONF_VECTOR_SEARCH,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_SEARCH,
    DOMAIN,
    DOMAIN_ATTRIBUTES,
    EXECUTE_SERVICES_TOOL,
)
from .embeddings import EntityEmbeddingIndex
from .ollama_client import OllamaError, chat as ollama_chat

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up HA Voice Agent conversation entity."""
    async_add_entities([HAVoiceAgentEntity(hass, config_entry)])


class HAVoiceAgentEntity(ConversationEntity, conversation.AbstractConversationAgent):
    """HA Voice Agent conversation entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Local",
            model=entry.options.get(CONF_MODEL, DEFAULT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._embed_index: EntityEmbeddingIndex | None = None

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """Build the embedding index after HA is ready."""
        await super().async_added_to_hass()
        opts = self.entry.options
        if opts.get(CONF_VECTOR_SEARCH, DEFAULT_VECTOR_SEARCH):
            self._embed_index = EntityEmbeddingIndex(
                hass=self.hass,
                ollama_url=opts.get(CONF_OLLAMA_URL, DEFAULT_OLLAMA_URL),
                embed_model=opts.get(CONF_EMBED_MODEL, DEFAULT_EMBED_MODEL),
                top_k=opts.get(CONF_TOP_K, DEFAULT_TOP_K),
            )
            # Build index from currently exposed entities (non-blocking)
            all_entities = self._get_all_exposed_entities()
            self.hass.async_create_task(
                self._embed_index.async_setup(all_entities)
            )

    async def async_will_remove_from_hass(self) -> None:
        """Clean up listeners."""
        if self._embed_index:
            self._embed_index.async_teardown()

    # ------------------------------------------------------------------
    # HA conversation agent entry point
    # ------------------------------------------------------------------

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a voice/text input."""
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._handle(user_input, chat_log)

    async def _handle(
        self, user_input: ConversationInput, chat_log: ChatLog
    ) -> ConversationResult:
        conversation_id = chat_log.conversation_id

        # --- Entity selection: vector search + area union ---
        opts = self.entry.options
        use_vector = opts.get(CONF_VECTOR_SEARCH, DEFAULT_VECTOR_SEARCH)

        if use_vector and self._embed_index and self._embed_index.is_ready:
            top_ids = await self._embed_index.async_search(user_input.text)
            area_ids = self._get_area_entity_ids(user_input.device_id)
            selected_ids = set(top_ids) | area_ids
            exposed_entities = self._get_entities_by_ids(selected_ids)
            _LOGGER.info(
                "Vector search: %d top + %d area = %d total entities (from %d)",
                len(top_ids),
                len(area_ids),
                len(exposed_entities),
                len(self._embed_index._entity_ids),
            )
        else:
            exposed_entities = self._get_all_exposed_entities()
            _LOGGER.info(
                "Vector search %s — using all %d entities",
                "disabled" if not use_vector else "index not ready",
                len(exposed_entities),
            )

        # Build messages
        system_prompt = self._render_prompt(exposed_entities, user_input)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        for msg in chat_log.content:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                if msg.role in ("user", "assistant") and msg.content:
                    messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_input.text})

        _LOGGER.info(
            "Query: %r  entities_in_prompt=%d  msgs=%d",
            user_input.text,
            len(exposed_entities),
            len(messages),
        )

        try:
            response_text = await self._query(user_input, messages, 0)
        except OllamaError as err:
            _LOGGER.error("Ollama error: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I couldn't reach the local AI: {err}",
            )
            return ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error("HA error: %s", err, exc_info=True)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)
        return ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    # ------------------------------------------------------------------
    # Ollama query + tool call loop
    # ------------------------------------------------------------------

    async def _query(
        self,
        user_input: ConversationInput,
        messages: list[dict[str, Any]],
        depth: int,
    ) -> str:
        opts = self.entry.options
        max_tool_calls = opts.get(CONF_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CALLS)

        if depth > max_tool_calls:
            return "I'm sorry, I ran into a problem completing that request."

        data = await ollama_chat(
            ollama_url=opts.get(CONF_OLLAMA_URL, DEFAULT_OLLAMA_URL),
            model=opts.get(CONF_MODEL, DEFAULT_MODEL),
            messages=messages,
            tools=[EXECUTE_SERVICES_TOOL],
            num_ctx=opts.get(CONF_NUM_CTX, DEFAULT_NUM_CTX),
            temperature=opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            num_predict=opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
        )

        msg = data.get("message", {})
        content: str = msg.get("content") or ""
        tool_calls: list[dict] = msg.get("tool_calls") or []

        if not tool_calls:
            return content.strip()

        messages.append(msg)

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            arguments = fn.get("arguments", {})

            if fn_name == "execute_services":
                result = await self._execute_services(arguments)
            else:
                _LOGGER.warning("Unknown tool: %s", fn_name)
                result = f"Unknown tool: {fn_name}"

            messages.append({"role": "tool", "content": str(result)})

        return await self._query(user_input, messages, depth + 1)

    async def _execute_services(self, arguments: dict) -> str:
        service_list = arguments.get("list", [])
        if not service_list:
            return "No services to execute."

        results = []
        for call in service_list:
            domain = call.get("domain", "")
            service = call.get("service", "")
            service_data: dict = dict(call.get("service_data", {}))

            if not domain or not service:
                results.append("Skipped: missing domain or service.")
                continue

            entity_id = service_data.pop("entity_id", None)
            target: dict[str, Any] = {}
            if entity_id:
                target["entity_id"] = entity_id

            _LOGGER.info(
                "Service call: %s.%s  target=%s  data=%s",
                domain, service, target, service_data,
            )
            try:
                await self.hass.services.async_call(
                    domain,
                    service,
                    service_data or {},
                    target=target or None,
                    blocking=True,
                )
                results.append(f"{domain}.{service}: OK")
            except HomeAssistantError as err:
                _LOGGER.error("Service %s.%s failed: %s", domain, service, err)
                results.append(f"{domain}.{service}: failed — {err}")

        return "; ".join(results)

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _get_all_exposed_entities(self) -> list[dict[str, Any]]:
        """Return all exposed entities with state + domain attributes."""
        entity_reg = er.async_get(self.hass)
        result = []
        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, conversation.DOMAIN, state.entity_id):
                continue
            entity = entity_reg.async_get(state.entity_id)
            domain = state.domain
            attr_keys = DOMAIN_ATTRIBUTES.get(domain, [])
            attrs = {k: v for k in attr_keys if (v := state.attributes.get(k)) is not None}
            result.append({
                "entity_id": state.entity_id,
                "name": state.name,
                "state": state.state,
                "attributes": attrs or None,
                "aliases": (entity.aliases or []) if entity else [],
            })
        return result

    def _get_entities_by_ids(self, entity_ids: set[str]) -> list[dict[str, Any]]:
        """Return entity dicts for the given entity_id set."""
        entity_reg = er.async_get(self.hass)
        result = []
        for eid in entity_ids:
            state = self.hass.states.get(eid)
            if state is None:
                continue
            if not async_should_expose(self.hass, conversation.DOMAIN, eid):
                continue
            entity = entity_reg.async_get(eid)
            domain = state.domain
            attr_keys = DOMAIN_ATTRIBUTES.get(domain, [])
            attrs = {k: v for k in attr_keys if (v := state.attributes.get(k)) is not None}
            result.append({
                "entity_id": eid,
                "name": state.name,
                "state": state.state,
                "attributes": attrs or None,
                "aliases": (entity.aliases or []) if entity else [],
            })
        return result

    def _get_area_entity_ids(self, device_id: str | None) -> set[str]:
        """Return entity_ids of all exposed entities in the same area as device_id."""
        if not device_id:
            return set()
        dev_reg = dr.async_get(self.hass)
        device = dev_reg.async_get(device_id)
        if not device or not device.area_id:
            return set()
        entity_reg = er.async_get(self.hass)
        area_entity_ids = {
            e.entity_id
            for e in entity_reg.entities.values()
            if e.area_id == device.area_id
            and async_should_expose(self.hass, conversation.DOMAIN, e.entity_id)
        }
        _LOGGER.debug(
            "Area entities for device %s (area %s): %d",
            device_id, device.area_id, len(area_entity_ids),
        )
        return area_entity_ids

    def _render_prompt(
        self,
        exposed_entities: list[dict],
        user_input: ConversationInput,
    ) -> str:
        raw = self.entry.options.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)
        try:
            return template.Template(raw, self.hass).async_render(
                {
                    "ha_name": self.hass.config.location_name,
                    "exposed_entities": exposed_entities,
                    "current_device_id": user_input.device_id,
                },
                parse_result=False,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("Error rendering system prompt: %s", err)
            return raw
