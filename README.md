# Transformer-Chatbot
Intro:

At the most basic level, a chatbot is a computer program that simulates and processes human conversation (either written or spoken), allowing humans to interact with digital devices as if they were communicating with a real person.
For this assignment, we will be building a chatbot that will be trained on Cornell movie dialogue corpus that can chat like movie characters.

Blog:
https://medium.com/@umarfaruk_56318/chatbot-using-tensorflow-trained-on-cornell-movie-data-set-e24a1111937d

Deployment Video:
https://youtu.be/Pzztrsnq-mw

## PocketSidekick MVP

This repository now includes a lightweight `PocketSidekick` MVP implementation with:

- Router-based intent handling (`chat` vs `/tool:<name>` commands)
- Built-in tool registry (`echo`, `word_count`)
- SQLite-backed conversational memory
- Persona mode for response shaping
- Quantization config export utility (`artifacts/quantization.json`)

### Quick start

```bash
python -m pocketsidekick.app
python -m unittest tests/test_pocketsidekick.py
```
