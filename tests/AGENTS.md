# Test Guidelines for Agents

This directory contains integration tests for `aquaregia`.

Treat the tests here as **public API and behavior contracts**. These tests should validate what a crate consumer can observe from outside the crate, not private implementation details.

AI assistance is welcome here, but do not add or modify tests you do not understand. If you cannot explain the behavior being asserted and why it matters, stop and gather more context first.

Keep this file principle-based. Do not turn it into an exhaustive checklist of test files, fixtures, or internal APIs; those drift faster than the code and make the guidance less trustworthy.

## 1. Choose the Right Test Location

- Put tests in `tests/` when they verify **public behavior** across module boundaries.
- Keep tests in `src/...` when they need access to **private helpers** or validate small internal logic in isolation.
- Do not move private/internal checks into `tests/` just to increase test count.

## 2. When You Must Add or Update Tests

Add or update tests whenever a change affects any of the following:

- Public request/response behavior.
- Public control flow or lifecycle behavior.
- Public error behavior.
- Public streaming behavior.
- Retry, timeout, or cancellation behavior.
- Public builder configuration or validation rules.
- Previously reported bugs or regressions.

If the change is user-visible and no test is added, there should be a concrete reason.

## 3. Coverage Surface Matters More Than Test Count

Do not optimize for "more tests." Optimize for **risk coverage**.

For a meaningful behavior change, cover the relevant surface:

- Happy path: prove the intended behavior works.
- Edge path: prove the important boundary or variant works.
- Failure path: prove invalid input, malformed provider data, or upstream errors are handled correctly.
- Regression path: if fixing a bug, reproduce the old failure mode and prove it no longer happens.

One well-targeted test is better than several shallow ones.

## 4. What Integration Tests Should Assert

Prefer assertions on externally visible outcomes:

- Returned responses, events, or errors.
- Request payload shape when serialization is part of the contract.
- Number of upstream requests when retry or multi-step behavior matters.
- Observable lifecycle order when callbacks or hooks are part of the feature.

Do not assert on incidental implementation details that consumers cannot observe.

## 5. Behavior Coverage Expectations

When a change affects a meaningful behavior surface, cover both sides of the contract when relevant:

- Outbound behavior: what is sent.
- Inbound behavior: what is returned or normalized.
- Error behavior: how failures are classified and exposed.
- Boundary behavior: EOF, partial data, empty data, invalid data, or stopping conditions.

If the same feature is implemented differently across different paths, do not assume one test is enough for all of them.

## 6. Streaming and Multi-Step Systems

For streaming or multi-step logic, prefer tests that validate:

- Ordering.
- Finalization.
- State transitions.
- Error propagation.
- Boundary conditions.

Streaming bugs often hide in boundary conditions. Test the boundary, not just the nominal stream.

## 7. Retry, Timeout, and Cancellation Tests

These behaviors are easy to claim and easy to under-test.

When touching them, assert observable behavior such as:

- The request is not sent when cancellation is already triggered.
- The operation stops after the expected number of attempts.
- Retryable vs non-retryable errors behave differently.
- `Retry-After` is honored when applicable.

Avoid tests that only check configuration builders if the runtime behavior is the real risk.

## 8. Determinism and Test Design

- Use `wiremock` for provider-facing integration tests.
- Keep tests deterministic and local. Do not rely on external networks or real API keys.
- Prefer small, purpose-built mock payloads over copied large payloads unless full payload fidelity is necessary.
- Keep each test focused on one behavior contract.
- Use descriptive test names that explain the expected behavior.

## 9. Avoid Low-Value Tests

Do not add tests that:

- Only duplicate coverage already provided by another integration test.
- Assert trivial builder chaining with no behavioral consequence.
- Snapshot large JSON payloads without asserting the important contract.
- Mirror implementation structure instead of user-visible behavior.

## 10. Practical Rule of Thumb

Before finishing a change in this repo, ask:

1. What user-visible behavior changed?
2. What is the most likely regression?
3. Which single integration test would catch that regression fastest?

Write that test first if the answer is clear.
