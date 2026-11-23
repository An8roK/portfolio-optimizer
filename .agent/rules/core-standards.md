---
trigger: always_on
---

# Core Project Standards

- **Plan first:** Briefly outline the approach and affected files before making multi-file or structural edits.
- **Keep code modular:** Break logic into small, single-purpose functions and reuse existing utilities instead of duplicating code.
- **Explain non-obvious logic:** Add brief comments explaining *why* a design choice or non-trivial step exists, not just what the syntax does.
- **Protect secrets:** Never hardcode API keys, passwords, or credentials; load them from environment variables (`.env`).
- **Verify changes:** Run relevant tests, linters, or sanity checks before marking a task complete.