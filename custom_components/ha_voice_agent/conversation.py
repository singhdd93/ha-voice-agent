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
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED, MATCH_ALL
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
    CONF_LLM_LOG_LEVEL,
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
    DEFAULT_LLM_LOG_LEVEL,
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
        """Build the embedding index after HA is fully started."""
        await super().async_added_to_hass()
        opts = self.entry.options
        if opts.get(CONF_VECTOR_SEARCH, DEFAULT_VECTOR_SEARCH):
            self._embed_index = EntityEmbeddingIndex(
                hass=self.hass,
                ollama_url=opts.get(CONF_OLLAMA_URL, DEFAULT_OLLAMA_URL),
                embed_model=opts.get(CONF_EMBED_MODEL, DEFAULT_EMBED_MODEL),
                top_k=opts.get(CONF_TOP_K, DEFAULT_TOP_K),
            )
            if self.hass.is_running:
                # HA already fully started (e.g. integration reload) — build now
                self.hass.async_create_task(
                    self._embed_index.async_setup(self._get_all_exposed_entities())
                )
            else:
                # Defer until all integrations have loaded their entities
                async def _on_ha_started(_event) -> None:
                    self.hass.async_create_task(
                        self._embed_index.async_setup(self._get_all_exposed_entities())
                    )

                self.hass.bus.async_listen_once(
                    EVENT_HOMEASSISTANT_STARTED, _on_ha_started
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

        # chat_log.content already includes the current user turn (added by async_get_chat_log).
        # Do NOT append user_input.text again — that causes a duplicate that confuses the model.
        for msg in chat_log.content:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                if msg.role in ("user", "assistant") and msg.content:
                    messages.append({"role": msg.role, "content": msg.content})

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
        force_no_tools: bool = False,
    ) -> str:
        opts = self.entry.options
        max_tool_calls = opts.get(CONF_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CALLS)

        if depth > max_tool_calls:
            return "I'm sorry, I ran into a problem completing that request."

        data = await ollama_chat(
            hass=self.hass,
            ollama_url=opts.get(CONF_OLLAMA_URL, DEFAULT_OLLAMA_URL),
            model=opts.get(CONF_MODEL, DEFAULT_MODEL),
            messages=messages,
            tools=None if force_no_tools else [EXECUTE_SERVICES_TOOL],
            num_ctx=opts.get(CONF_NUM_CTX, DEFAULT_NUM_CTX),
            temperature=opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            num_predict=opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            log_level=opts.get(CONF_LLM_LOG_LEVEL, DEFAULT_LLM_LOG_LEVEL),
        )

        msg = data.get("message", {})
        content: str = msg.get("content") or ""
        tool_calls: list[dict] = msg.get("tool_calls") or []
        done_reason: str = data.get("done_reason", "stop")

        if done_reason == "length":
            _LOGGER.warning(
                "Response truncated at %d tokens — increase max_tokens if this is frequent",
                data.get("eval_count", 0),
            )

        if not tool_calls:
            return content.strip() or "Sorry, I didn't get a complete response. Please try again."

        # If every tool call is a state query (no control actions), the model is trying
        # to use tools to look up state it already has in its context. Resolve the state
        # programmatically and inject it, then do a final no-tools call.
        all_query = all(
            tc.get("function", {}).get("arguments", {}).get("service", "") in self._QUERY_SERVICE_NAMES
            for tc in tool_calls
        )
        if all_query and not force_no_tools:
            state_text = self._try_direct_state_lookup(user_input.text)
            _LOGGER.info(
                "Model made %d query-only tool call(s) — direct lookup: %s",
                len(tool_calls), state_text or "no match",
            )
            if state_text:
                # Inject assistant tool-call msg + synthetic result, then answer without tools
                messages.append(msg)
                for _ in tool_calls:
                    messages.append({"role": "tool", "content": state_text})
            return await self._query(user_input, messages, depth + 1, force_no_tools=True)

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

    # Service names that look like state queries — not real HA services.
    _QUERY_SERVICE_NAMES = frozenset({"state", "get", "get_state", "status", "check", "query", "info"})

    async def _execute_services(self, arguments: dict) -> str:
        import ast
        import json as _json

        service_list = arguments.get("list", [])
        if isinstance(service_list, str):
            try:
                service_list = _json.loads(service_list)
            except Exception:
                service_list = []

        # Model sometimes passes domain/service/service_data at the top level instead of in a list.
        if not service_list and "domain" in arguments and "service" in arguments:
            service_list = [arguments]

        if not service_list:
            return "No services to execute."

        results = []
        for call in service_list:
            domain = call.get("domain", "")
            service = call.get("service", "")
            raw_data = call.get("service_data", {})

            # Parse service_data if the model sent it as a string (Python dict literal or JSON).
            if isinstance(raw_data, str):
                try:
                    raw_data = _json.loads(raw_data)
                except Exception:
                    try:
                        raw_data = ast.literal_eval(raw_data)
                    except Exception:
                        raw_data = {}

            # service_data may be a list of entity_ids (model bug) — normalise to dict.
            if isinstance(raw_data, list):
                service_data: dict = {"entity_id": raw_data[0] if len(raw_data) == 1 else raw_data}
            elif isinstance(raw_data, dict):
                service_data = dict(raw_data)
            else:
                service_data = {}

            if not domain or not service:
                results.append("Skipped: missing domain or service.")
                continue

            # When model calls a query-like service (fan.state, fan.get, etc.),
            # look up the real entity state and return it — don't error.
            if service in self._QUERY_SERVICE_NAMES:
                raw_eid = service_data.get("entity_id", "")
                results.append(self._lookup_entity_state(domain, raw_eid))
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

    # Keyword → HA domain for direct state lookup
    _DOMAIN_KEYWORDS: dict[str, str] = {
        "fan": "fan",
        "light": "light",
        "lamp": "light",
        "ac": "climate",
        "aircon": "climate",
        "aircondition": "climate",
        "airconditioner": "climate",
        "tv": "remote",
        "television": "remote",
        "speaker": "media_player",
        "switch": "switch",
        "blind": "cover",
        "curtain": "cover",
    }

    def _try_direct_state_lookup(self, user_text: str) -> str | None:
        """Resolve a state query by matching area name + device type from the user text.

        Returns a formatted state string if a match is found, else None.
        """
        area_reg = ar.async_get(self.hass)
        entity_reg = er.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)

        # Normalise query for substring matching (strip spaces, dashes, underscores)
        def _norm(s: str) -> str:
            return s.lower().replace(" ", "").replace("-", "").replace("_", "")

        query_norm = _norm(user_text)

        # Find which HA areas appear in the query
        matched_areas: list[tuple[str, str]] = []  # (area_id, area_name)
        for area in area_reg.async_list_areas():
            if _norm(area.name) in query_norm:
                matched_areas.append((area.id, area.name))

        if not matched_areas:
            return None

        # Find which device domain the query refers to
        target_domain: str | None = None
        for keyword, domain in self._DOMAIN_KEYWORDS.items():
            if keyword in query_norm:
                target_domain = domain
                break

        # Collect matching exposed entities
        results: list[str] = []
        for area_id, area_name in matched_areas:
            for entry in entity_reg.entities.values():
                # Resolve entity area (entity level → device level)
                eid_area = entry.area_id
                if not eid_area and entry.device_id:
                    device = dev_reg.async_get(entry.device_id)
                    if device:
                        eid_area = device.area_id

                if eid_area != area_id:
                    continue
                if not async_should_expose(self.hass, conversation.DOMAIN, entry.entity_id):
                    continue

                entity_domain = entry.entity_id.split(".")[0]
                if target_domain and entity_domain != target_domain:
                    continue

                state = self.hass.states.get(entry.entity_id)
                if state is None:
                    continue

                attr_keys = DOMAIN_ATTRIBUTES.get(entity_domain, [])
                attrs = {k: v for k in attr_keys if (v := state.attributes.get(k)) is not None}
                summary = f"{entry.entity_id} ({state.name}, area: {area_name}) is {state.state}"
                if attrs:
                    summary += f" | {attrs}"
                results.append(summary)

        if not results:
            return None

        return "; ".join(results)

    def _lookup_entity_state(self, domain: str, raw_entity_id: str | list) -> str:
        """Resolve a (possibly malformed) entity reference and return its state as text."""
        # Normalise list to first element.
        if isinstance(raw_entity_id, list):
            raw_entity_id = raw_entity_id[0] if raw_entity_id else ""

        candidates: list[str] = []
        if raw_entity_id:
            # 1. Try as-is.
            candidates.append(str(raw_entity_id))
            # 2. Prepend domain if no dot present.
            if "." not in str(raw_entity_id):
                candidates.append(f"{domain}.{raw_entity_id}")

        for eid in candidates:
            state = self.hass.states.get(eid)
            if state:
                attr_keys = DOMAIN_ATTRIBUTES.get(state.domain, [])
                attrs = {k: v for k in attr_keys if (v := state.attributes.get(k)) is not None}
                text = f"{eid} is {state.state}"
                if attrs:
                    text += f" | {attrs}"
                _LOGGER.info("State query resolved: %s → %s", eid, text)
                return text

        # Fallback: fuzzy search across all states in the same domain.
        needle = str(raw_entity_id).lower().replace("-", "_").replace(" ", "_")
        for state in self.hass.states.async_all(domain):
            if needle in state.entity_id:
                attr_keys = DOMAIN_ATTRIBUTES.get(state.domain, [])
                attrs = {k: v for k in attr_keys if (v := state.attributes.get(k)) is not None}
                text = f"{state.entity_id} is {state.state}"
                if attrs:
                    text += f" | {attrs}"
                _LOGGER.info("State query fuzzy-matched %r → %s", raw_entity_id, text)
                return text

        _LOGGER.warning("State query: could not resolve entity %r in domain %r", raw_entity_id, domain)
        return f"Could not find entity '{raw_entity_id}' in domain '{domain}'."

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _get_all_exposed_entities(self) -> list[dict[str, Any]]:
        """Return all exposed entities with state + domain attributes + area name."""
        entity_reg = er.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)
        result = []
        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, conversation.DOMAIN, state.entity_id):
                continue
            entity = entity_reg.async_get(state.entity_id)
            domain = state.domain
            attr_keys = DOMAIN_ATTRIBUTES.get(domain, [])
            attrs = {k: v for k in attr_keys if (v := state.attributes.get(k)) is not None}
            area_name = self._resolve_area_name(entity, dev_reg, area_reg)
            result.append({
                "entity_id": state.entity_id,
                "name": state.name,
                "state": state.state,
                "attributes": attrs or None,
                "aliases": (entity.aliases or []) if entity else [],
                "area": area_name,
            })
        return result

    def _get_entities_by_ids(self, entity_ids: set[str]) -> list[dict[str, Any]]:
        """Return entity dicts for the given entity_id set."""
        entity_reg = er.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)
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
            area_name = self._resolve_area_name(entity, dev_reg, area_reg)
            result.append({
                "entity_id": eid,
                "name": state.name,
                "state": state.state,
                "attributes": attrs or None,
                "aliases": (entity.aliases or []) if entity else [],
                "area": area_name,
            })
        return result

    def _resolve_area_name(
        self,
        entity: er.RegistryEntry | None,
        dev_reg: dr.DeviceRegistry,
        area_reg: ar.AreaRegistry,
    ) -> str | None:
        """Return area name for an entity (entity area → device area → None)."""
        if entity is None:
            return None
        area_id = entity.area_id
        if not area_id and entity.device_id:
            device = dev_reg.async_get(entity.device_id)
            if device:
                area_id = device.area_id
        if not area_id:
            return None
        area = area_reg.async_get_area(area_id)
        return area.name if area else None

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
