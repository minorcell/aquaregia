---
name: writing-proposals
description: Use when drafting, rewriting, reviewing, or standardizing a technical proposal, RFC, design issue, product-change proposal, API proposal, language proposal, tooling proposal, or implementation plan that should be incremental, simple, subtractive, boundary-aware, and grounded in real user stories.
---

# Writing Proposals

## Overview

Write proposals as small, reviewable design increments. The goal is not to describe a whole system at once; the goal is to isolate the next necessary design decision, express it simply, remove avoidable complexity, define boundaries, and ground the change in real user stories.

A good proposal makes a reviewer able to decide, an implementer able to build, and a maintainer able to verify. Format is secondary to this thinking model.

## Core Thinking Model

### 1. Design Progressively

One proposal should settle one clear increment. Do not try to explain every related subsystem, future extension, or complete architecture in a single proposal.

Use follow-up proposals for adjacent decisions. Prefer a narrow proposal that can be accepted, implemented, and tested over a broad proposal that tries to solve the whole future.

### 2. Prefer Simple, Direct Expression

Choose the expression a reader can understand immediately. If one realistic example explains the design, lead with the example before abstract definitions. If one rule can explain the behavior, do not introduce multiple mechanisms.

The design should be short, teachable, implementable, and easy to remember.

### 3. Do Subtractive Design

The proposal should reduce redundancy or accidental complexity. Look for opportunities to remove:

- boilerplate code
- manual steps
- repeated configuration
- exposed intermediate mechanisms
- special cases users must remember
- duplicated concepts across similar workflows

Do not add features for completeness. Add the smallest design that lets the system absorb repeated work or remove unnecessary user burden.

### 4. Define Boundaries Carefully

Make the limits explicit:

- what this proposal solves
- what it does not solve
- which cases are valid
- which cases are errors
- whether existing behavior changes
- what workaround exists today
- where this design meets existing mechanisms

Clear boundaries prevent a proposal from expanding into a large system design.

### 5. Abstract From Broad User Stories

Do not add a feature mechanically from one case. Look across related user stories and find the shared need underneath them. The proposal should capture the smallest common design that serves those stories.

If the need appears in only one narrow case, keep the proposal narrow. If multiple stories point to the same missing expression, define that expression directly.

## Workflow

1. Collect the concrete user stories, examples, bugs, or workflows that motivate the change.
2. Extract the common need and state it in one sentence.
3. Identify the smallest design increment that addresses that need.
4. Remove scope that belongs in future proposals.
5. Express the proposed behavior with examples, rules, and boundaries.
6. Show current workaround and why the new design removes redundancy.
7. Add compatibility, errors, alternatives, and tests only to the depth required by risk.

## Template Selection

| Situation | Use |
|---|---|
| Seed idea, narrow behavior, link to upstream design, or small issue | Short template |
| Normal technical change for review | Standard template |
| Language, API, CLI, protocol, architecture, data model, or user-visible workflow with lasting behavior | Full specification |
| DSL, query language, or dense notation | Full specification plus Scope and Quick Reference |

Increase template weight only when risk increases. Do not use a full specification to make a simple proposal look more important.

## Short Template

Use this when the design is small and the decision point is obvious.

```md
### Proposal

State the change directly. Include the concrete new behavior, API, command, rule, or user-visible result.

### Background

Explain why this is needed. Link related issues, upstream designs, examples, existing behavior, or user stories.

### Workarounds

Describe the current workaround and why it is insufficient. If none exists, write `none`.
```

For compatibility or adoption proposals, add:

```md
### Reference

Link the upstream accepted design and summarize the local implication in one sentence.
```

## Standard Template

Use this for most proposals.

```md
# Proposal: <specific change>

## 1. Summary

One paragraph: what changes, where it applies, and the smallest design increment being proposed.

## 2. User Stories / Motivation

List the concrete stories or workflows. Then state the common need they reveal.

## 3. Current Workaround

Describe how users solve this today and what redundancy, manual work, or inconsistency remains.

## 4. Goals

- Goal 1
- Goal 2
- Goal 3

## 5. Out of Scope

- What this proposal intentionally does not solve
- Future work for separate proposals
- Related systems not changed here

## 6. Proposal

### 6.1 Design Rule

State the core rule or invariant.

### 6.2 Syntax / API / Interface

Show the exact user-facing or implementer-facing contract.

### 6.3 Examples as Specification

Use examples to define behavior, not as decoration.

```go
// current form
...

// proposed form
...

// equivalent / desugared form
...
```

### 6.4 Boundary Cases

Show valid, invalid, omitted, conflict, or fallback cases.

## 7. Error Handling

| Condition | Behavior |
|---|---|
| Invalid input | Error behavior |
| Missing optional dependency | Warning / skip behavior |
| Conflict | Fatal behavior |

## 8. Compatibility

State whether the change is additive, breaking, opt-in, migrated, or version-gated.

## 9. Alternatives Considered

### 9.1 <Alternative>

Why it was considered and why it was rejected.

## 10. Testing Strategy

| Test case | Method |
|---|---|
| Normal case | Expected assertion |
| Error case | Expected diagnostic |
| Compatibility case | Existing behavior unchanged |

## 11. Summary of Changes

| Area | Change |
|---|---|
| Parser / API / CLI / UI / docs | Specific change |
```

## Full Specification Additions

Add only the sections that create review value.

| Proposal type | Add details |
|---|---|
| Language or syntax | Grammar, desugaring, type rules, valid and invalid examples, formatter/tooling impact |
| API or library | Signature, parameters, return values, side effects, ownership, concurrency, error model |
| CLI or toolchain | Command interface, flags, trigger points, execution order, path resolution, generated files |
| Data model or protocol | Schema, field meanings, validation, versioning, migration, compatibility |
| DSL or query language | Scope, Quick Reference, Core Concepts, syntax rules, query examples, implementation considerations |
| Architecture | Module boundaries, ownership, dependency direction, global side effects, rollback path |
| Product workflow | User states, permissions, empty/error/loading states, audit or analytics behavior if needed |

## Style Rules

- Lead with the smallest decision point, not the whole future system.
- Prefer one clear rule over multiple mechanisms.
- Use code examples as normative specification: current form, proposed form, equivalent/desugared form, and invalid cases.
- Put user stories before abstract motivation when the need comes from usage.
- Treat `Workarounds` as evidence for why the proposal should exist.
- Treat `Out of Scope` as mandatory for non-trivial changes.
- Prefer tables for parameters, flags, errors, behavior matrices, and summaries.
- Keep implementation notes practical: parser, type checker, API, command, docs, tests.
- Put uncertain decisions in `Open Questions`; do not silently hard-code them.
- Avoid marketing language and broad architecture claims.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Trying to describe the whole system | Narrow to the next design increment |
| Adding a feature from one case | Check broader user stories and extract the shared need |
| Overexplaining before examples | Show the concrete behavior first |
| Creating a new mechanism too early | Reuse or simplify existing principles where possible |
| Missing boundary cases | Add valid, invalid, conflict, and fallback examples |
| Treating examples as illustrative only | Make examples define normative behavior |
| Ignoring workaround quality | Explain what redundancy or manual work remains today |
| Full spec for a tiny change | Use the short template unless risk justifies more |

## Review Checklist

Before delivering a proposal, verify:

- It advances one clear increment.
- The design is as simple and direct as the problem allows.
- It removes redundancy, manual work, or special-case memory.
- User stories justify the need beyond one accidental case.
- Boundaries, invalid cases, and out-of-scope items are explicit.
- Examples define the behavior precisely enough to test.
- Compatibility and migration are stated.
- The proposal can become an implementation checklist without expanding scope.
