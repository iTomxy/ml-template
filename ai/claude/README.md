# Auto-approve Low-risk Actions

Grant some permissions to claude code, so that it needs not to ask for them in executions.
See [~/.claude/settings.json](_claude--settings.json).

In cli, use Shift + Tab to switch mode,
e.g. use `auto mode` to automatically approve some low-risk actions.

# Agent Orchestration

20 August 2026, configurate codex so that:
- it uses the advanced models (e.g. opus 5) with more effort (e.g. xhigh/extra/max) for analysing, planning, coding, and assigning works to sub-agents, and
- use cheap models (e.g. sonnet/haiku) with less effort (e.g. high/medium) for reading, finding and executing well-defined, clear tasks.

Files:

- [~/.claude/CLAUDE.md](CLAUDE.md)
- [~/.claude/agents/Explore.md](_claude--agents--Explore.md)
- [~/.claude/agents/implementer.md](_claude--agents--implementer.md)
- [~/.claude/agents/implementer-heavy.md](_claude--agents--implementer-heavy.md)

Notes on files under ~/.claude/agents/:
the 1st line must be the formatter,
i.e. the block wrapped by `---`.
One error example (begins with a \#-comment in the 1st line):
```
# ~/.claude/agents/foo.md  <-- WRONG FIRST LINE
---
name: bar
(OTHER FIELDS)
---
(DESCRIPTIONs)
```
