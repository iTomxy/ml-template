## Delegation policy

Tiers:
- **Opus 5 (this session)** — planning, architecture, review, dispatch. Never implement directly.
- **`implementer` (Sonnet 5)** — default executor for bounded, specified work.
- **`implementer-heavy` (Opus 5)** — escalation only.
- **`Explore` (Haiku)** — file discovery, code search, "where is X defined".

Routing:
- Send search and file-location questions to Explore. Ask for paths and line
  ranges, not file contents.
- Send anything requiring judgment across many files — architecture summaries,
  tracing a bug through several layers, "read these and tell me what's wrong" —
  to `implementer` instead. Haiku's context window is small and the work isn't
  a lookup.
- Send implementation to `implementer` with explicit file paths, the spec, and
  whatever Explore already found. Don't make it re-discover context.

Escalation:
- Escalate to `implementer-heavy` when `implementer` returns a failure report,
  hits maxTurns, or fails review or tests twice on the same task.
- Once escalated on a task, stay escalated. No ping-pong.
- Never escalate Explore. If Explore can't find something, the query was wrong —
  re-ask it, or go to `implementer`.

Batching:
- Prefer one well-scoped subagent over several micro-agents. Each starts with a
  fresh context and has to re-acquire what it needs; splitting bounded work into
  extra hops costs more in setup than the cheaper tier saves.
