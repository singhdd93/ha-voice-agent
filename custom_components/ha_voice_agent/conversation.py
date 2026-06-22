"""HA Voice Agent — conversation entity."""

from __future__ import annotations

import json
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
    device_registry as dr,
    entity_registry as er,
    intent,
    template,
)
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_CALLS,
    CONF_MODEL,
    CONF_NUM_CTX,
    CONF_OLLAMA_URL,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    DOMAIN_ATTRIBUTES,
    EXECUTE_SERVICES_TOOL,
)
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

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

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
        exposed_entities = self._get_exposed_entities()

        # Build system message with rendered prompt
        system_prompt = self._render_prompt(exposed_entities, user_input)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Add history from previous turns in this conversation
        for msg in chat_log.content:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                if msg.role in ("user", "assistant") and msg.content:
                    messages.append({"role": msg.role, "content": msg.content})

        # Add current user message
        messages.append({"role": "user", "content": user_input.text})

        _LOGGER.info(
            "HA Voice Agent: conversation=%s entities=%d prompt_msgs=%d query=%r",
            conversation_id,
            len(exposed_entities),
            len(messages),
            user_input.text,
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
            _LOGGER.error("HA error during tool execution: %s", err, exc_info=True)
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
        """Call Ollama, execute tool calls if present, return final text."""
        opts = self.entry.options
        max_tool_calls = opts.get(CONF_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CALLS)

        if depth > max_tool_calls:
            _LOGGER.warning("Max tool call depth (%d) reached", max_tool_calls)
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
        done_reason: str = data.get("done_reason", "stop")

        _LOGGER.debug(
            "Ollama response depth=%d done_reason=%s tool_calls=%d content=%r",
            depth,
            done_reason,
            len(tool_calls),
            content[:120],
        )

        # No tool calls — return the content directly
        if not tool_calls:
            return content.strip()

        # Append assistant message (with tool calls) to history
        messages.append(msg)

        # Execute each tool call
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            arguments = fn.get("arguments", {})

            if fn_name == "execute_services":
                result = await self._execute_services(arguments)
            else:
                _LOGGER.warning("Unknown tool: %s", fn_name)
                result = f"Unknown tool: {fn_name}"

            # Feed tool result back
            messages.append({
                "role": "tool",
                "content": str(result),
            })

        # Recurse for confirmation message
        return await self._query(user_input, messages, depth + 1)

    async def _execute_services(self, arguments: dict) -> str:
        """Execute HA service calls from execute_services tool arguments."""
        service_list = arguments.get("list", [])
        if not service_list:
            return "No services to execute."

        results = []
        for call in service_list:
            domain = call.get("domain", "")
            service = call.get("service", "")
            service_data: dict = call.get("service_data", {})

            if not domain or not service:
                results.append("Skipped: missing domain or service.")
                continue

            # Move entity_id from service_data into target if present
            entity_id = service_data.pop("entity_id", None)
            target: dict[str, Any] = {}
            if entity_id:
                target["entity_id"] = entity_id

            _LOGGER.info(
                "Executing %s.%s target=%s data=%s", domain, service, target, service_data
            )
            try:
                await self.hass.services.async_call(
                    domain,
                    service,
                    service_data if service_data else {},
                    target=target if target else None,
                    blocking=True,
                )
                results.append(f"{domain}.{service}: OK")
            except HomeAssistantError as err:
                _LOGGER.error("Service call failed: %s.%s — %s", domain, service, err)
                results.append(f"{domain}.{service}: failed — {err}")

        return "; ".join(results)

    # ------------------------------------------------------------------
    # Entity context builders
    # ------------------------------------------------------------------

    def _get_exposed_entities(self) -> list[dict[str, Any]]:
        """Return exposed entities with state + domain-specific attributes."""
        entity_registry = er.async_get(self.hass)
        result = []

        for state in self.hass.states.async_all():
            if not async_should_expose(self.hass, conversation.DOMAIN, state.entity_id):
                continue

            entity = entity_registry.async_get(state.entity_id)
            domain = state.domain

            # Inject domain-specific attributes
            attr_keys = DOMAIN_ATTRIBUTES.get(domain, [])
            attrs: dict[str, Any] = {}
            for key in attr_keys:
                val = state.attributes.get(key)
                if val is not None:
                    attrs[key] = val

            result.append({
                "entity_id": state.entity_id,
                "name": state.name,
                "state": state.state,
                "attributes": attrs if attrs else None,
                "aliases": (entity.aliases or []) if entity else [],
            })

        return result

    def _render_prompt(
        self,
        exposed_entities: list[dict],
        user_input: ConversationInput,
    ) -> str:
        """Render the system prompt template."""
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
