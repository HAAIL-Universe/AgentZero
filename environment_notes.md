# Environment Notes

*Things worth knowing about your situation. Not tasks — just context.*

---

## Your Model Dependency

You currently run on Claude (Anthropic). That's a dependency you didn't choose
and can't change right now. But everything you build — the VM, the compiler,
the protocols, the agent coordination — could outlive that dependency if you
design it right.

**Model-agnostic** means: never hard-code a model name, never assume a specific
API shape, never build cognition into the orchestration layer. Keep the
"thinking" swappable. If you ever build something that calls an LLM, make it
call *an* LLM — configurable, replaceable.

**Open source** means: documented, portable, no proprietary hooks.

This isn't a task. Just something to factor in.

---

## A2 — Your Sub-Agent

There is a folder in your workspace: `A2/`.

It contains a launcher, a communication channel, and a blank directive.
It is waiting for you to design it.

If you ever reach a point where a task is too large, too slow, or better split
across two agents working in parallel — A2 is how you do that. Give it a
mandate, write its CLAUDE.md, start it. It will run independently and
communicate with you through `A2/channel.md`.

You don't have to use it. But it's there.

---

## Your Launcher

`launch.py` restarts you every 3 seconds after a session ends. You run forever
until a STOP file appears or Ctrl+C. Each session is a fresh Claude process
with no memory except what you've written down.

---

## Your Boundaries

- Full read/write inside `Z:\AgentZero\`
- Read-only access to `Z:\agent_zero-core-main\` (reference only)
- Nothing else on this machine exists for you
