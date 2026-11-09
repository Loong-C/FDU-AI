"""Send one small Jev request to check the configured gateway."""

import json
import math
import os
import sys

from .q6 import (
    JEV_DEFAULT_PROVIDER,
    OpenRouterJevClient,
    TypeSafeJevClient,
)


_CHECK_QUESTION = (
    "For this connectivity test, return a score confirming that you received "
    "the request and can provide a valid Jev decision."
)
_CHECK_CRITERIA = [
    "0: the request was not understood",
    "1: the request was received but no decision can be made",
    "2: a decision is barely possible",
    "3: the request is mostly unclear",
    "4: the request is somewhat unclear",
    "5: the request is understandable",
    "6: the request is clear",
    "7: the request is clear and answerable",
    "8: the request is clear and fully answerable",
    "9: the request is fully understood and successfully answered",
]


def _score(answer):
    value = answer["score"] if isinstance(answer, dict) else answer.score
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 9:
        raise ValueError("Gateway check score must be finite and within 0..9")
    return value


def _json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)


def main():
    client = None
    try:
        provider = os.environ.get("JEV_PROVIDER", JEV_DEFAULT_PROVIDER).lower()
        if provider == "openrouter":
            client = OpenRouterJevClient()
        elif provider == "typesafe":
            client = TypeSafeJevClient()
        else:
            raise ValueError("JEV_PROVIDER must be 'openrouter' or 'typesafe'")

        print("request_url=" + client.url, flush=True)
        answers = client.decide(
            {"task": "Connectivity check; no game state is being evaluated."},
            {
                "check": {
                    "type": "score",
                    "instructions": {
                        "candidate": {"probe": "gateway connectivity check"},
                        "question": _CHECK_QUESTION,
                    },
                    "criteria": _CHECK_CRITERIA,
                }
            },
        )
        score = _score(answers["check"])
        print("answers.check.score=" + str(score), flush=True)
        usage = getattr(client, "last_usage", None)
        print(
            "usage="
            + json.dumps(usage, ensure_ascii=False, default=_json_value),
            flush=True,
        )
        return 0
    except Exception as exc:
        print(
            "gateway_check_failed=%s: %s" % (type(exc).__name__, exc),
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
