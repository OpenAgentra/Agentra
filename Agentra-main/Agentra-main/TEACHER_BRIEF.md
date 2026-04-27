# Agentra Teacher Brief

## Core Thesis

Agentra is not just a browser automation demo. It is an early backend/runtime foundation for human-supervised computer-use agents that can browse, use tools, keep per-thread context, and expose an audit trail.

The project goal is to explore this question:

> What does a trustworthy and controllable autonomous agent runtime need, beyond "the model can click buttons"?

---

## Preliminary Findings

### 1. Computer-use agents are real, but still unreliable

Official products and research previews show that computer-use is already useful, but still far from fully reliable.

- OpenAI presents Computer-Using Agent (CUA) as a research preview and makes clear that it is still early.
- OpenAI also reports benchmark progress, but human performance remains much higher on full OS tasks.

Source:

- [OpenAI Computer-Using Agent](https://openai.com/index/computer-using-agent/)
- [OpenAI Operator System Card](https://openai.com/index/operator-system-card/)

Why this matters for Agentra:

- The opportunity is real.
- A serious project should not claim "fully autonomous magic."
- The right focus is runtime safety, observability, and controlled autonomy.

### 2. Human oversight is a product requirement, not an optional extra

Leading systems emphasize confirmation and oversight for risky actions.

- OpenAI notes that user confirmation is needed for sensitive actions such as login details or CAPTCHA-like situations.
- Anthropic's computer-use documentation also frames these systems as agent loops that need careful supervision in practice.

Source:

- [OpenAI Computer-Using Agent](https://openai.com/index/computer-using-agent/)
- [Anthropic Computer Use](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)

Why this matters for Agentra:

- Approval policy is not a polish feature.
- Approval gates should be part of the runtime architecture.
- Asking the user before risky actions is a key research and engineering finding.

### 3. Shared browser control is important for usable agents

The jump from "agent runs alone" to "agent and user can operate the same session" is a meaningful product difference.

- Browser Use documents human takeover as a first-class workflow.
- This validates the importance of shared session control instead of disconnected manual testing tools.

Source:

- [Browser Use Human Takeover](https://docs.browser-use.com/tips/live-view/human-takeover)

Why this matters for Agentra:

- Shared browser session is a strong direction for the project.
- User takeover should happen on the same live session the agent is using.

### 4. Memory is still incomplete in existing agent products

Current products often remember chat/search context better than actual agent actions.

- Perplexity's Comet memory help page says agent actions are not stored in memory.

Source:

- [Perplexity Comet Memory](https://comet-help.perplexity.ai/en/articles/12658438-comet-memory)

Why this matters for Agentra:

- There is room to differentiate with:
  - thread-local working memory
  - long-term searchable memory
  - screenshot/action/history retrieval

### 5. The real research gap is controllable autonomy

The most interesting technical question is not "can an agent click?" but:

- how do we isolate tasks?
- how do we pause/resume them?
- how do we approve risky actions?
- how do we audit what happened?
- how do we let a human take over?

This is where Agentra is strongest as a university project: it explores runtime design, not only prompt tricks.

---

## What Agentra Already Demonstrates

Current prototype directions already map to the findings above:

- `thread-aware runtime`
  - multiple independent task threads
  - per-thread state and run history
- `approval policy engine`
  - agent works autonomously, but risky actions can require permission
- `shared browser session foundation`
  - manual actions and agent actions can converge on the same browser session model
- `memory architecture`
  - thread working memory plus longer-term memory direction
- `audit/report trail`
  - runs can be replayed and inspected instead of being black-box behavior
- `live operator UI`
  - useful as a testing and demonstration surface for runtime behavior

---

## Honest Limitations

These should be stated clearly in the presentation:

- This is an MVP / prototype, not a production-grade agent.
- Full reliability is not solved.
- Latency and visual smoothness are still weaker than a native browser/desktop stream.
- Human takeover UX is only partially implemented.
- The strongest contribution right now is the backend architecture direction, not a finished end-user app.

This honesty makes the project sound more professional, not less.

---

## Suggested 5-Minute Presentation Flow

### 1. Problem

"Computer-use agents are becoming real, but they are still unreliable and risky when acting without supervision."

### 2. Preliminary Findings

Use 3 short findings:

1. Current agent systems are promising but still early.
2. Human approval and takeover are essential.
3. Memory and auditability are still weak in many current products.

### 3. Project Claim

"So instead of building only a flashy automation demo, I started building a controllable agent runtime."

### 4. Demo

Show:

- entering a task
- the agent using the browser
- live TV/report style monitoring
- if possible, one approval or thread/runtime behavior

### 5. Conclusion

"My project explores what a trustworthy autonomous agent architecture should look like: thread-aware, approval-aware, auditable, and eventually human-takeover capable."

---

## Recommended Demo Story

Choose a safe browser-first task such as:

> Open a site, inspect content, take a screenshot, summarize what is visible, and finish cleanly.

Avoid tomorrow:

- logins
- payments
- CAPTCHAs
- destructive file actions
- complex desktop control

The demo should support the thesis, not create risk.

---

## One-Sentence Research Positioning

If your teacher asks what the project is really about, answer with this:

> Agentra is an exploration of how to build autonomous agents that are not only capable, but also controllable, inspectable, and safe enough to collaborate with a human operator.

---

## Short Q&A Answers

### "What is your preliminary finding?"

My preliminary finding is that the key challenge in autonomous agents is not only model capability, but runtime control: approvals, takeover, memory, auditability, and isolation between tasks.

### "What is new in your prototype?"

The prototype focuses on a thread-aware agent runtime with approval flow, live monitoring, audit/reporting, and a path toward shared human-agent browser control.

### "What is next?"

The next milestone is to strengthen the backend core further: better shared browser control, stronger memory retrieval, safer approval rules, and a desktop shell on top of the same runtime.
