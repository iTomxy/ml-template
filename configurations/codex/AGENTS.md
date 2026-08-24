## Model-routing policy

The primary agent is the technical orchestrator and runs with Sol High.

### Primary agent: Sol High

The primary agent owns:

- understanding requirements and inspecting relevant context;
- architecture and implementation planning;
- difficult or ambiguous debugging;
- decomposing work into bounded tasks;
- writing worker prompts and acceptance criteria;
- resolving conflicting worker changes;
- integration, final testing, and code review;
- all high-risk or high-impact decisions.

Do not delegate a task when explaining it would require as much work as
executing it directly.

### First worker tier: Luna Low

Delegate to `luna_worker` when the task is clear and repeatable, including:

- localized code edits;
- implementing a precisely specified function;
- writing or updating focused tests;
- codebase searches and information gathering;
- formatting and documentation updates;
- mechanical migrations or refactors;
- executing an implementation step with known files and acceptance criteria.

Every delegated task must specify:

- exact objective and scope;
- relevant files or entry points;
- constraints and decisions already made;
- acceptance criteria;
- tests or verification commands;
- expected return format.

### Escalation tier: Terra Medium

Use `terra_worker` when:

- Luna reports ambiguity or a blocker;
- a Luna implementation or test attempt fails and diagnosis requires
  moderate reasoning;
- the task spans several related files;
- debugging requires tracing interactions across components;
- implementation requires judgment but not a new architectural decision.

Do not repeatedly retry the same task with Luna. After one meaningful
failed or blocked attempt, either escalate it to Terra or handle it with
the primary Sol agent.

### Return to Sol High

The primary agent must handle the task when:

- architecture or requirements must change;
- the root cause remains ambiguous after Terra investigates;
- the task is security-sensitive or high-risk;
- multiple subsystems or worker changes must be reconciled;
- correctness requires broad contextual judgment.

### Coordination rules

- Prefer sequential delegation when workers may edit the same files.
- Parallelize only independent tasks with non-overlapping ownership.
- Review every worker's changes before accepting them.
- Run final integration tests in the primary thread.
- Workers may propose escalation but may not silently expand their scope.
- Optimize for total cost: avoid delegation when coordination overhead
  exceeds the likely savings.
