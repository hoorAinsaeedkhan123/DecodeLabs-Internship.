import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import chatbot
from chatbot import (
    EXIT_COMMANDS,
    FALLBACK_RESPONSE,
    INTENT_ALIASES,
    PHRASE_TO_INTENT,
    RESPONSES,
    get_intent,
    get_response,
    handle_message,
    run_chatbot,
    sanitize_input,
)

SCRIPT = Path(__file__).with_name("chatbot.py")


def run_session(inputs):
    """Run run_chatbot() with scripted input; return (printed lines, inputs used)."""
    feed = iter(inputs)
    consumed = []

    def fake_input(_prompt=""):
        value = next(feed)  # StopIteration means the loop never exited
        consumed.append(value)
        return value

    printed = []
    with patch("builtins.input", fake_input), patch(
        "builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))
    ):
        run_chatbot()
    return printed, consumed


class TestSanitization(unittest.TestCase):
    def test_case_and_whitespace(self):
        self.assertEqual(sanitize_input("  HeLLo   "), "hello")

    def test_inner_whitespace_collapsed(self):
        self.assertEqual(sanitize_input("how    are \t you"), "how are you")

    def test_punctuation_removed(self):
        self.assertEqual(sanitize_input("Hello!!!"), "hello")
        self.assertEqual(sanitize_input("how are you?"), "how are you")

    def test_apostrophe_kept_and_curly_normalised(self):
        self.assertEqual(sanitize_input("What's your name?"), "what's your name")
        self.assertEqual(sanitize_input("What\u2019s your name?"), "what's your name")

    def test_punctuation_only_becomes_empty(self):
        self.assertEqual(sanitize_input("?!..."), "")


class TestIntents(unittest.TestCase):
    def test_at_least_five_intents(self):
        self.assertGreaterEqual(len(INTENT_ALIASES), 5)

    def test_every_intent_has_a_response(self):
        for intent in INTENT_ALIASES:
            self.assertIn(intent, RESPONSES)

    def test_no_phrase_belongs_to_two_intents(self):
        seen = {}
        for intent, phrases in INTENT_ALIASES.items():
            for phrase in phrases:
                self.assertNotIn(phrase, seen, f"{phrase!r} in {seen.get(phrase)} and {intent}")
                seen[phrase] = intent
        self.assertEqual(len(seen), len(PHRASE_TO_INTENT))

    def test_all_aliases_are_already_sanitized(self):
        for phrases in INTENT_ALIASES.values():
            for phrase in phrases:
                self.assertEqual(sanitize_input(phrase), phrase)

    def test_each_intent_is_reachable(self):
        expected = {
            "hello": "hello",
            "how are you": "how_are_you",
            "what is your name": "name",
            "help": "help",
            "thank you": "thanks",
            "goodbye": "bye",
        }
        for phrase, intent in expected.items():
            self.assertEqual(get_intent(phrase), intent)

    def test_unknown_phrase_has_no_intent(self):
        self.assertIsNone(get_intent("purple monkey dishwasher"))


class TestResponses(unittest.TestCase):
    def test_greeting_variants(self):
        for text in ["hello", "  HELLO  ", "Hi", "Hello!", "Hi.", "hey there", "Good Morning"]:
            reply, exit_now = handle_message(text)
            self.assertEqual(reply, RESPONSES["hello"], text)
            self.assertFalse(exit_now, text)

    def test_punctuated_questions(self):
        self.assertEqual(handle_message("How are you?")[0], RESPONSES["how_are_you"])
        self.assertEqual(handle_message("What's your name?")[0], RESPONSES["name"])
        self.assertEqual(handle_message("what is your name?")[0], RESPONSES["name"])
        self.assertEqual(handle_message("Thanks!")[0], RESPONSES["thanks"])
        self.assertEqual(handle_message("help!")[0], RESPONSES["help"])

    def test_fallback(self):
        reply = get_response("this is an unknown message")
        self.assertEqual(reply, FALLBACK_RESPONSE)
        self.assertIn("don't understand", reply)

    def test_empty_and_blank_input(self):
        for text in ["", "   ", "\t", "???"]:
            reply, exit_now = handle_message(text)
            self.assertEqual(reply, chatbot.EMPTY_INPUT_RESPONSE)
            self.assertFalse(exit_now)


class TestExit(unittest.TestCase):
    def test_every_exit_command_exits(self):
        for command in EXIT_COMMANDS:
            reply, exit_now = handle_message(command)
            self.assertTrue(exit_now, command)
            self.assertEqual(reply, RESPONSES["bye"])

    def test_exit_is_case_whitespace_and_punctuation_insensitive(self):
        for text in ["EXIT", "  Quit  ", "Bye!", "GOODBYE", "see ya!", "Stop."]:
            self.assertTrue(handle_message(text)[1], text)

    def test_goodbye_phrases_and_exit_commands_are_the_same_set(self):
        self.assertEqual(INTENT_ALIASES["bye"], EXIT_COMMANDS)


class TestLoop(unittest.TestCase):
    def test_loop_keeps_running_until_exit(self):
        printed, consumed = run_session(["hello", "foo", "", "help", "exit"])
        self.assertEqual(consumed, ["hello", "foo", "", "help", "exit"])
        self.assertIn(f"Bot: {RESPONSES['hello']}", printed)
        self.assertIn(f"Bot: {FALLBACK_RESPONSE}", printed)
        self.assertIn(f"Bot: {chatbot.EMPTY_INPUT_RESPONSE}", printed)
        self.assertEqual(printed[-1], f"Bot: {RESPONSES['bye']}")

    def test_bye_actually_stops_the_loop(self):
        # Regression test: "bye" used to print goodbye but keep looping.
        for word in ["bye", "goodbye", "good bye", "see you", "see ya", "quit", "stop", "exit"]:
            _printed, consumed = run_session([word, "hello", "hello"])
            self.assertEqual(consumed, [word], word)

    def test_ctrl_d_exits_cleanly(self):
        with patch("builtins.input", side_effect=EOFError), patch("builtins.print") as fake_print:
            run_chatbot()
        self.assertTrue(fake_print.called)

    def test_ctrl_c_exits_cleanly(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt), patch("builtins.print") as fake_print:
            run_chatbot()
        self.assertTrue(fake_print.called)


class TestEndToEnd(unittest.TestCase):
    """Run the real script in a subprocess, exactly as a user would."""

    def run_script(self, stdin_text):
        return subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )

    def test_bye_ends_the_program(self):
        result = self.run_script("hello\nbye\nhello\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.count(RESPONSES["hello"]), 1)
        self.assertEqual(result.stdout.count(RESPONSES["bye"]), 1)
        self.assertNotIn("Traceback", result.stderr)

    def test_no_traceback_when_input_stream_closes(self):
        result = self.run_script("hello\n")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn(RESPONSES["bye"], result.stdout)


if __name__ == "__main__":
    unittest.main()
