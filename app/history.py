"""Approximate token counting for trimming message history."""


def approx_tokens(messages: list) -> int:
    """Estimates token count as ~4 chars per token."""
    return sum(len(str(m.content)) // 4 for m in messages)
