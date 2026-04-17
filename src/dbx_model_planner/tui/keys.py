"""Cross-platform raw keyboard input for the TUI."""

from __future__ import annotations

import sys

# Key constants
KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_ENTER = "enter"
KEY_ESCAPE = "escape"
KEY_TAB = "tab"
KEY_BACKSPACE = "backspace"
KEY_PAGE_UP = "page_up"
KEY_PAGE_DOWN = "page_down"
KEY_HOME = "home"
KEY_END = "end"


def read_key() -> str:
    """Read a single keypress. Returns a string representing the key.

    Returns key constants (KEY_UP, KEY_DOWN, etc.) for special keys,
    or the character itself for printable keys.
    """
    if sys.platform == "win32":
        return _read_key_windows()
    else:
        return _read_key_unix()


def read_key_nonblocking(timeout: float = 0.15) -> str | None:
    """Read a single keypress with a timeout.

    Returns ``None`` if no key is pressed within *timeout* seconds.
    This allows background tasks to run while still accepting user input.
    """
    if sys.platform == "win32":
        return _read_key_windows_nonblocking(timeout)
    else:
        return _read_key_unix_nonblocking(timeout)


def _read_key_windows() -> str:
    import msvcrt

    ch = msvcrt.getwch()

    # Special keys: msvcrt returns \x00 or \xe0 followed by another byte
    if ch in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        return {
            "H": KEY_UP,
            "P": KEY_DOWN,
            "K": KEY_LEFT,
            "M": KEY_RIGHT,
            "I": KEY_PAGE_UP,
            "Q": KEY_PAGE_DOWN,
            "G": KEY_HOME,
            "O": KEY_END,
        }.get(code, "")

    if ch == "\r":
        return KEY_ENTER
    if ch == "\x1b":
        return KEY_ESCAPE
    if ch == "\t":
        return KEY_TAB
    if ch == "\x08":
        return KEY_BACKSPACE
    return ch


def _read_key_unix() -> str:
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return KEY_UP
            if seq == "[B":
                return KEY_DOWN
            if seq == "[C":
                return KEY_RIGHT
            if seq == "[D":
                return KEY_LEFT
            if seq == "[5":
                # Page Up: ESC [ 5 ~
                sys.stdin.read(1)  # consume '~'
                return KEY_PAGE_UP
            if seq == "[6":
                # Page Down: ESC [ 6 ~
                sys.stdin.read(1)  # consume '~'
                return KEY_PAGE_DOWN
            if seq == "[H":
                return KEY_HOME
            if seq == "[F":
                return KEY_END
            return KEY_ESCAPE

        if ch == "\r" or ch == "\n":
            return KEY_ENTER
        if ch == "\t":
            return KEY_TAB
        if ch == "\x7f":
            return KEY_BACKSPACE
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_key_windows_nonblocking(timeout: float) -> str | None:
    import msvcrt
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            return _read_key_windows()
        time.sleep(0.01)
    return None


def _read_key_unix_nonblocking(timeout: float) -> str | None:
    import select
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None

        ch = sys.stdin.read(1)

        if ch == "\x1b":
            # Check for escape sequence with a short timeout
            ready2, _, _ = select.select([fd], [], [], 0.05)
            if ready2:
                seq = sys.stdin.read(2)
                if seq == "[A":
                    return KEY_UP
                if seq == "[B":
                    return KEY_DOWN
                if seq == "[C":
                    return KEY_RIGHT
                if seq == "[D":
                    return KEY_LEFT
                if seq == "[5":
                    sys.stdin.read(1)  # consume '~'
                    return KEY_PAGE_UP
                if seq == "[6":
                    sys.stdin.read(1)  # consume '~'
                    return KEY_PAGE_DOWN
                if seq == "[H":
                    return KEY_HOME
                if seq == "[F":
                    return KEY_END
            return KEY_ESCAPE

        if ch in ("\r", "\n"):
            return KEY_ENTER
        if ch == "\t":
            return KEY_TAB
        if ch == "\x7f":
            return KEY_BACKSPACE
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
