"""Constants for HA Voice Agent."""

DOMAIN = "ha_voice_agent"

# Config entry keys
CONF_OLLAMA_URL = "ollama_url"
CONF_MODEL = "model"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOOL_CALLS = "max_tool_calls"
CONF_NUM_CTX = "num_ctx"
CONF_EMBED_MODEL = "embed_model"
CONF_TOP_K = "top_k"
CONF_VECTOR_SEARCH = "vector_search"

# Defaults
DEFAULT_OLLAMA_URL = "http://10.5.6.50:11434"
DEFAULT_MODEL = "nemotron-3-nano-ha:latest"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOOL_CALLS = 3
DEFAULT_NUM_CTX = 4096
DEFAULT_EMBED_MODEL = "nomic-embed-text-v2-moe:latest"
DEFAULT_TOP_K = 15
DEFAULT_VECTOR_SEARCH = True

DEFAULT_SYSTEM_PROMPT = """\
You are a smart home voice assistant controlling Home Assistant at {{ ha_name }}.
Current time: {{ now().strftime('%H:%M, %A %d %B %Y') }}

Available devices:
{% for entity in exposed_entities -%}
- {{ entity.entity_id }} | {{ entity.name }} | {{ entity.state }}{% if entity.attributes %} | {{ entity.attributes }}{% endif %}

{% endfor %}

Rules:
1. Always include the full entity_id with domain prefix (e.g., light.bedroom, fan.guest_bedroom_fan).
2. Use turn_on / turn_off — never toggle.
3. Fan speed: domain=fan, service=set_percentage, data={percentage: X}. Valid steps: 17/33/50/67/83/100.
4. The domain in every service call must match the entity_id prefix exactly.
5. Never call execute_services to query state — read state from Available devices above.
6. Only use standard Home Assistant service names.
7. For TV power: use remote.turn_on / remote.turn_off on remote.* entities.
8. After executing any action, always respond with a brief confirmation sentence.
9. If asked about device state, answer directly from Available devices — do not call a service.\
"""

# Domain → attributes to inject into context
DOMAIN_ATTRIBUTES: dict[str, list[str]] = {
    "fan": ["percentage", "preset_mode"],
    "climate": [
        "current_temperature",
        "temperature",
        "hvac_mode",
        "fan_mode",
        "preset_mode",
        "hvac_modes",
    ],
    "light": ["brightness", "color_temp_kelvin"],
    "media_player": ["volume_level", "source", "media_title"],
    "cover": ["current_position"],
    "remote": [],
    "switch": [],
    "sensor": [],
    "binary_sensor": [],
}

# execute_services tool definition (Ollama native format)
EXECUTE_SERVICES_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_services",
        "description": (
            "Execute one or more Home Assistant service calls to control devices. "
            "Use this for all device control actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "HA service domain (must match entity_id prefix)",
                            },
                            "service": {
                                "type": "string",
                                "description": "Service name, e.g. turn_on, turn_off, set_percentage",
                            },
                            "service_data": {
                                "type": "object",
                                "description": (
                                    "Service data. Include entity_id as a string "
                                    "or list of strings."
                                ),
                            },
                        },
                        "required": ["domain", "service", "service_data"],
                    },
                }
            },
            "required": ["list"],
        },
    },
}
