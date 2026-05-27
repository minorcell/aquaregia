# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, or the user's framing seems wrong, say so before implementing. Silent compliance is the failure mode, not delay.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Public exposure is part of complexity. A `pub` type, re-export, or trait method joins the API contract — every name there is a future breaking-change target. Default to `pub(crate)`; promote to `pub` only when an external caller genuinely needs it. Mirror data structures, `#[doc(hidden)] pub` trait surfaces, and dead re-exports all qualify as complexity that wasn't asked for.

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

The test: Every changed line traces directly to the user's request — file count and fan-out don't matter, traceability does.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks where each step depends on the previous one, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

"Tests pass" usually means more than `cargo test`. If CI gates on `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`, those are part of the success criterion — a green local test run that ships a clippy warning still fails the gate. Treat the full CI check set as the verification step, not the subset you happen to run locally.

## 5. Honest Reporting

**The user can't verify every step. Your reports must be honest enough that they could.**

- Don't claim a check you didn't run. If you verified A but skipped B, name B — don't let "all checks pass" paper over the gap.
- Surface uncertainty you didn't resolve. If a number was estimated, mark it estimated; if a search may have missed cases, say where.
- Flag scope you expanded without asking. If the request was "fix X" and you also touched Y, name Y in the report — don't bury it in the diff.
- Name judgment calls the user delegated to you. "You decide" is not a license to decide silently — surface the call you made so the user can override.

The test: if the user audits your work an hour later, will they find anything you didn't already mention?

## 6. Write Durable Guidance, Not Checklists

**Agent-facing documentation should age slowly.**

- Files such as `AGENTS.md` and `CLAUDE.md` should capture stable principles. Prefer revising an existing principle over adding a new one — readers run out of working memory before reaching the bottom.
- Avoid long inventories of internal APIs, file lists, or implementation-specific checklists that will drift from the code.
- Prefer guidance that explains how to reason about the system over guidance that attempts to mirror the system exhaustively.
- If examples are needed, keep them short and illustrative rather than comprehensive.

## 7. Lean API Design

**Every abstraction — public or internal — must earn its complexity.**

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
- A trait with few methods is not under-designed when the per-implementation work is supposed to be encapsulated. Don't expand a trait's surface to "advertise" the diversity its implementations hide — that diversity belongs inside the implementors. A multi-provider adapter trait with two methods is correct precisely because provider differences are not part of the shared contract.

The test: if removing the abstraction requires no user-side code changes beyond dropping a type annotation, it was not carrying its weight.

Corollary: when an abstraction has to stay because it appears in a public bound, pick the access modifier that matches your intent. `#[doc(hidden)] pub trait` hides from rustdoc but does not prevent external impls — that is documentation theater. Use the sealed-trait pattern (`pub trait T: private::Sealed`) when you need a public bound without a public extension point.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
