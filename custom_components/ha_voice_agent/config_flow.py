"""Config flow for HA Voice Agent."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback

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
)
from .ollama_client import test_connection


class HAVoiceAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle config flow for HA Voice Agent."""

    VERSION = 1

    async def async_step_user(self, user_input=None) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            ok = await test_connection(
                self.hass, user_input[CONF_OLLAMA_URL], user_input[CONF_MODEL]
            )
            if not ok:
                errors["base"] = "cannot_connect"
            else:
                await self.async_set_unique_id(
                    f"{user_input[CONF_OLLAMA_URL]}_{user_input[CONF_MODEL]}"
                )
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"HA Voice Agent ({user_input[CONF_MODEL]})",
                    data={},
                    options=user_input,
                )

        schema = vol.Schema(
            {
                vol.Required(CONF_OLLAMA_URL, default=DEFAULT_OLLAMA_URL): str,
                vol.Required(CONF_MODEL, default=DEFAULT_MODEL): str,
            }
        )
        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry) -> HAVoiceAgentOptionsFlow:
        return HAVoiceAgentOptionsFlow()


class HAVoiceAgentOptionsFlow(OptionsFlow):
    """Handle options for HA Voice Agent."""

    async def async_step_init(self, user_input=None) -> ConfigFlowResult:
        opts = self.config_entry.options

        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_OLLAMA_URL, default=opts.get(CONF_OLLAMA_URL, DEFAULT_OLLAMA_URL)
                ): str,
                vol.Required(
                    CONF_MODEL, default=opts.get(CONF_MODEL, DEFAULT_MODEL)
                ): str,
                vol.Optional(
                    CONF_NUM_CTX, default=opts.get(CONF_NUM_CTX, DEFAULT_NUM_CTX)
                ): int,
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                ): int,
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): float,
                vol.Optional(
                    CONF_MAX_TOOL_CALLS,
                    default=opts.get(CONF_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CALLS),
                ): int,
                vol.Optional(
                    CONF_VECTOR_SEARCH,
                    default=opts.get(CONF_VECTOR_SEARCH, DEFAULT_VECTOR_SEARCH),
                ): bool,
                vol.Optional(
                    CONF_EMBED_MODEL,
                    default=opts.get(CONF_EMBED_MODEL, DEFAULT_EMBED_MODEL),
                ): str,
                vol.Optional(
                    CONF_TOP_K,
                    default=opts.get(CONF_TOP_K, DEFAULT_TOP_K),
                ): int,
                vol.Optional(
                    CONF_SYSTEM_PROMPT,
                    default=opts.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT),
                ): str,
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
