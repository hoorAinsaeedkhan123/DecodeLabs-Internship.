# DecodeLabs AI Internship — Project 1
## Rule-Based AI Chatbot 🤖

### Project objective
Build a simple rule-based chatbot that responds to predefined user inputs using explicit programmatic rules (control flow and decision-making logic, no machine learning).

### Features
- Continuous `while True` input loop
- Input sanitization: case, leading/trailing/extra whitespace, punctuation, curly apostrophes
- Dictionary-based knowledge base with **6 intents**
- O(1) intent lookup through a flat `phrase -> intent` dictionary
- `.get()` fallback response for anything the bot doesn't understand
- Clean exit: `exit`, `quit`, `stop`, `bye`, `goodbye`, `good bye`, `see you`, `see ya`
- Graceful shutdown on `Ctrl+C` / `Ctrl+D` (no traceback)
- Empty or punctuation-only input is handled without crashing
- Deterministic, explainable responses (a "white box": input -> logic -> output)

### Technology
- Python 3.9+
- No external packages required

### How to run

```bash
python chatbot.py
```

(On some systems use `python3 chatbot.py`.)

### How to run the tests

```bash
python -m unittest -v
```

The suite has 24 tests covering sanitization, every intent, the fallback, every exit command, the loop itself, `Ctrl+C` / `Ctrl+D`, and end-to-end runs of the real script.

### Example

```text
🤖 Rule-Based AI Chatbot
Type 'help' for supported commands or 'exit' to quit.

You: Hello!
Bot: Hi there! 👋 How can I help you today?

You: what can you do
Bot: I can respond to greetings, basic questions ('how are you', 'what is your name'), and thanks. Type 'exit' or 'bye' to quit.

You: something random
Bot: I'm sorry, I don't understand that yet. Try 'help' to see what I can respond to.

You: bye
Bot: Goodbye! Have a great day! 👋
```

### Supported intents
| # | Intent | Example inputs |
|---|---|---|
| 1 | Greeting | `hello`, `hi`, `hey there`, `good morning` |
| 2 | How the bot is doing | `how are you`, `how's it going` |
| 3 | Bot identity / name | `what is your name`, `who are you` |
| 4 | Help / capabilities | `help`, `what can you do` |
| 5 | Thanks | `thanks`, `thank you`, `thx` |
| 6 | Goodbye / exit | `bye`, `goodbye`, `exit`, `quit`, `stop` |

### Architecture (IPO model)

| Phase | Function | Responsibility |
|---|---|---|
| Input | `sanitize_input()` | Lower-case, strip punctuation, collapse whitespace |
| Process | `get_intent()` | O(1) dictionary lookup of the sanitized phrase |
| Output | `get_response()` / `handle_message()` | Response lookup with fallback; returns `(reply, should_exit)` |
| Loop | `run_chatbot()` | `while True` cycle with a clean `break` on exit |

`handle_message()` contains no `input()` or `print()`, which is what makes the logic easy to unit test.

### Project requirement mapping

| Requirement (from the project brief) | Implementation |
|---|---|
| Handle greetings | `INTENT_ALIASES["hello"]` |
| Handle exit commands | `EXIT_COMMANDS` + `break` in `run_chatbot()` |
| Use if-else logic | Empty-input, exit and `should_exit` checks in `handle_message()` / `run_chatbot()` |
| Run in a continuous loop | `while True` in `run_chatbot()` |
| Input loop | `run_chatbot()` |
| Sanitization (case and whitespace) | `sanitize_input()` |
| Knowledge base (dictionary, 5+ intents) | `INTENT_ALIASES`, `RESPONSES`, `PHRASE_TO_INTENT` (6 intents) |
| Fallback for unknown input | `RESPONSES.get(intent, FALLBACK_RESPONSE)` |
| Exit strategy (clean break) | `EXIT_COMMANDS`, plus `Ctrl+C` / `Ctrl+D` handling |

### Known limitations (by design)
The bot uses **exact matching** on sanitized phrases, so it understands the phrases listed in `INTENT_ALIASES` and nothing more. For example, `hello how are you` in one message is not recognized. Semantic matching is the natural next step (Project 2) and is deliberately out of scope for a rule-based foundation project.

### Design notes
The chatbot deliberately uses deterministic rules rather than an LLM or machine-learning model, because Project 1 is intended to establish the fundamentals of control flow, decision-making logic, exact input mapping, and continuous interaction. A flat dictionary replaces a long `if / elif` ladder, so lookup time stays constant however many rules are added.

The goodbye intent and `EXIT_COMMANDS` share one set of phrases, so "saying goodbye" and "leaving" can never drift apart.

### Suggested portfolio description

> Built a deterministic rule-based AI chatbot in Python as part of the DecodeLabs AI Internship. Implemented input sanitization, O(1) dictionary-based intent mapping, continuous conversation flow, clean exit handling, and fallback responses, backed by a 24-test unit suite, to demonstrate foundational AI and control-flow concepts.
