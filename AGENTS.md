# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Use AI Responsibly

**AI is allowed. Judgment is not outsourced.**

- AI assistance is allowed and encouraged for exploration, drafting, and implementation.
- The person or agent making the change must still understand what changed, why it changed, and how it was validated.
- Do not ship code, tests, or documentation that you cannot explain, review, or debug.
- Use AI to accelerate execution, not to replace engineering judgment.

## 6. Write Durable Guidance, Not Checklists

**Agent-facing documentation should age slowly.**

- Files such as `AGENTS.md` and `CLAUDE.md` should capture stable principles, constraints, and decision rules.
- Avoid long inventories of internal APIs, file lists, or implementation-specific checklists that will drift from the code.
- Prefer guidance that explains how to reason about the system over guidance that attempts to mirror the system exhaustively.
- If examples are needed, keep them short and illustrative rather than comprehensive.

## 7. Lean API Design

**Type-level machinery must earn its complexity.**

Every generic parameter, marker trait, phantom type, or feature flag that ends up in user-facing code costs the user cognitive overhead every time they write a type annotation or read an error message. Add that cost only when there is a concrete, demonstrable benefit.

Before adding type-level structure, ask:

- Does this prevent a real class of mistakes that a clear runtime error would not catch?
- Does this enable real code reuse — not just the appearance of a unified interface?
- Can a user realistically misuse the API without this constraint, and would the failure be silent?

If the answers are "no", the abstraction is decorative. Remove it.

Concrete rules for this codebase:

- No phantom type parameters on public structs unless they enforce a boundary that the construction site cannot enforce.
- No feature flags that gate no code (`openai = []` is noise, not a feature).
- No trait hierarchies where each implementor does something entirely different — four `impl`s with no shared runtime behaviour is not polymorphism, it is a dispatch table in disguise.
- No extra wrapper types when a plain value type (`String`, a struct with named fields) communicates the same contract.

The test: if removing the abstraction requires no user-side code changes beyond dropping a type annotation, it was not carrying its weight.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
