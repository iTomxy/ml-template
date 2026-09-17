# Auto-approve Low-risk Actions

Use `/permissions` to let codex approve low-rise actions without human in the loop.
Can also manually set it by adding `approvals_reviewer = "auto_review"` to [~/.codex/config.toml](_codex--config.toml).

# Agent Orchestration

20 August 2026, configurate codex so that:
- it uses the advanced models (e.g. GPT-5.6 Sol) with more effort (e.g. ultra or high) for analysing, planning, coding, and assigning works to sub-agents, and
- use cheap models (e.g. GPT-5.6 Luna/Terra) with less effort (e.g. medium or low) for reading, finding and executing well-defined, clear tasks.

Files:

- [AGENTS.md](AGENTS.md): can be global (~/.codex/AGENTS.md) or project-specific (PROJECT_PATH/AGENTS.md)
- [~/.codex/config.toml](_codex--config.toml)
- [~/.codex/agents/luna-worker.toml](_codex--agents--luna-worker.toml)
- [~/.codex/agents/terra-worker.tom](_codex--agents--terra-worker.toml)
