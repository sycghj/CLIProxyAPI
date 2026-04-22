import json
import os
import sys
import uuid

from anthropic import Anthropic


BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "http://100.108.152.121:8317")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "2rN12tFfCdmpKznfvb2gxOlvmWaW1ovxrKVghxfV")
MODEL = os.getenv("ANTHROPIC_MODEL", "gpt-5.4-high")
MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "20"))
SYSTEM_REPEAT = int(os.getenv("ANTHROPIC_SYSTEM_REPEAT", "500"))


def build_system_text() -> str:
    unique = f"cache-test-{uuid.uuid4()}"
    return ((f"SYSTEM-CACHE-BLOCK {unique} " + ("X" * 32) + " ") * SYSTEM_REPEAT).strip()


def to_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return json.loads(json.dumps(value, default=lambda o: getattr(o, "__dict__", str(o))))


def call_messages(client: Anthropic, system_text: str, user_text: str):
    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        ],
    )


def verdict(call1_usage: dict, call2_usage: dict) -> str:
    c1_create = int(call1_usage.get("cache_creation_input_tokens", 0) or 0)
    c1_read = int(call1_usage.get("cache_read_input_tokens", 0) or 0)
    c2_create = int(call2_usage.get("cache_creation_input_tokens", 0) or 0)
    c2_read = int(call2_usage.get("cache_read_input_tokens", 0) or 0)

    if c2_read > 0:
        return "PASS: second request hit cache"
    if c1_create > 0 and c1_read == 0 and c2_create > 0 and c2_read == 0:
        return "PARTIAL: cache markers likely passed through, but second request did not show a cache read"
    return "FAIL: no cache hit observed"


def main() -> int:
    client = Anthropic(api_key=API_KEY, base_url=BASE_URL)
    system_text = build_system_text()

    try:
        resp1 = call_messages(client, system_text, "call1")
        resp2 = call_messages(client, system_text, "call2")
    except Exception as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    usage1 = to_dict(resp1.usage)
    usage2 = to_dict(resp2.usage)

    result = {
        "base_url": BASE_URL,
        "model": MODEL,
        "call1_id": getattr(resp1, "id", None),
        "call1_usage": usage1,
        "call2_id": getattr(resp2, "id", None),
        "call2_usage": usage2,
        "verdict": verdict(usage1, usage2),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
