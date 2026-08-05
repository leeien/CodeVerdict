"""The one exception type the use-case layer raises."""

from __future__ import annotations


class CoreError(Exception):
    """A refused operation, with a reason fit to show a person.

    ``status`` carries HTTP-shaped semantics so neither front end has to invent
    its own error taxonomy: 404 not found, 409 wrong state for this operation,
    403 refused by policy, 502 an external tool (git/gh/an LLM) failed.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message

    def __str__(self) -> str:  # what the terminal prints
        return self.message


def not_found(what: str) -> CoreError:
    return CoreError(404, f"{what} not found")


def conflict(message: str) -> CoreError:
    return CoreError(409, message)


def refused(message: str) -> CoreError:
    return CoreError(403, message)


def upstream(message: str) -> CoreError:
    return CoreError(502, message)
