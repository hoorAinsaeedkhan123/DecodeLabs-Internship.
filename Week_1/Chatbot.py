"""
DecodeLabs AI Internship - Project 1
Rule-Based AI Chatbot

A deterministic, console-based chatbot built with Python.

Architecture (IPO model from the project brief):
    INPUT    -> sanitize_input()   normalise case, whitespace, punctuation
    PROCESS  -> get_intent()       O(1) dictionary lookup (phrase -> intent)
    OUTPUT   -> handle_message()   response lookup with a fallback
    LOOP     -> run_chatbot()      continuous `while True` cycle with a clean exit
"""

import re
import sys

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

# Every phrase that ends the conversation (the "kill command").
EXIT_COMMANDS = {
    "exit", "quit", "stop",
    "bye", "goodbye", "good bye", "see you", "see ya",
}

# intent -> phrases that trigger it (all phrases are stored already sanitized).
INTENT_ALIASES = {
    "hello": {
        "hello", "hi", "hey", "hey there", "hi there", "yo",
        "good morning", "good afternoon", "good evening",
    },
    "how_are_you": {
        "how are you", "how are u", "how r u", "how are you doing",
        "how's it going", "hows it going", "how is it going", "how are things",
    },
    "name": {
        "what is your name", "what's your name", "whats your name",
        "your name", "who are you", "tell me your name",
    },
    "help": {
        "help", "what can you do", "what do you do", "commands",
    },
    "thanks": {
        "thanks", "thank you", "thank u", "thx", "thanks a lot",
        "thank you so much", "many thanks",
    },
    # The goodbye intent shares its phrases with EXIT_COMMANDS so that
    # "saying goodbye" and "leaving" can never get out of sync.
    "bye": EXIT_COMMANDS,
}

RESPONSES = {
    "hello": "Hi there! \U0001F44B How can I help you today?",
    "how_are_you": "I'm doing well, thank you! \U0001F916 How are you?",
    "name": "I'm a rule-based AI chatbot built for DecodeLabs Project 1.",
    "help": (
        "I can respond to greetings, basic questions ('how are you', "
        "'what is your name'), and thanks. Type 'exit' or 'bye' to quit."
    ),
    "thanks": "You're welcome! \U0001F60A",
    "bye": "Goodbye! Have a great day! \U0001F44B",
}

FALLBACK_RESPONSE = (
    "I'm sorry, I don't understand that yet. "
    "Try 'help' to see what I can respond to."
)

EMPTY_INPUT_RESPONSE = "Please enter a message."

# Flat lookup table: sanitized phrase -> intent. Built once at start-up so
# every user message is resolved with a single O(1) dictionary access instead
# of scanning through every intent.
PHRASE_TO_INTENT = {
    phrase: intent
    for intent, phrases in INTENT_ALIASES.items()
    for phrase in phrases
}


# ---------------------------------------------------------------------------
# Phase 1: input & sanitization
# ---------------------------------------------------------------------------

def sanitize_input(user_input: str) -> str:
    """Normalise raw input so it can be matched reliably.

    - lower-cases the text
    - converts curly apostrophes to straight ones
    - removes punctuation (apostrophes are kept: "what's")
    - trims and collapses whitespace
    """
    text = user_input.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = text.replace("_", " ")
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Phase 2: process (intent matching)
# ---------------------------------------------------------------------------

def get_intent(user_input: str):
    """Return the intent for an already-sanitized message, or None."""
    return PHRASE_TO_INTENT.get(user_input)


def is_exit_command(user_input: str) -> bool:
    """True when an already-sanitized message should end the conversation."""
    return user_input in EXIT_COMMANDS


# ---------------------------------------------------------------------------
# Phase 3: output (response generation)
# ---------------------------------------------------------------------------

def get_response(user_input: str) -> str:
    """Return the reply for an already-sanitized message (fallback if unknown)."""
    return RESPONSES.get(get_intent(user_input), FALLBACK_RESPONSE)


def handle_message(raw_message: str) -> tuple[str, bool]:
    """Run one full IPO cycle.

    Returns (reply, should_exit). Keeping this free of input()/print() makes
    the chatbot logic easy to unit test.
    """
    user_input = sanitize_input(raw_message)

    if not user_input:
        return EMPTY_INPUT_RESPONSE, False
    if is_exit_command(user_input):
        return RESPONSES["bye"], True
    return get_response(user_input), False


# ---------------------------------------------------------------------------
# The heartbeat: continuous loop
# ---------------------------------------------------------------------------

def run_chatbot() -> None:
    """Run the chatbot until the user enters an exit command."""
    print("\U0001F916 Rule-Based AI Chatbot")
    print("Type 'help' for supported commands or 'exit' to quit.\n")

    while True:
        try:
            raw_message = input("You: ")
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C / closed input stream: leave politely, no traceback.
            print(f"\nBot: {RESPONSES['bye']}")
            break

        reply, should_exit = handle_message(raw_message)
        print(f"Bot: {reply}")

        if should_exit:
            break


def _configure_console() -> None:
    """Avoid UnicodeEncodeError for emoji on consoles that are not UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


if __name__ == "__main__":
    _configure_console()
    run_chatbot()
