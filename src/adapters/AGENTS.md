# Model Adapter Guidelines for Agents

This directory contains provider-specific adapter logic.

Treat this layer as a **compatibility boundary**, not a product surface for inventing new behavior. Its job is to translate between Aquaregia's unified types and each provider's documented API as faithfully as possible.

## 1. Align to Official Provider Semantics

- Changes in this directory must stay tightly aligned with the provider's official API behavior and documented wire format.
- Prefer the provider's canonical field names, event shapes, finish reasons, usage semantics, and error meanings.
- Do not replace documented provider behavior with guessed behavior just because it feels cleaner.
- If the provider API is ambiguous, prefer the most conservative interpretation and document the assumption in code comments or tests when necessary.

## 2. The Adapter Is a Translator, Not a Re-Designer

- Normalize provider behavior into Aquaregia's public abstractions, but do not silently invent semantics that the upstream API does not provide.
- Keep provider-specific quirks contained to the adapter layer.
- Do not leak one provider's conventions into another provider's adapter.
- If a capability is not truly cross-provider, do not force symmetry through fragile hacks.

## 3. Do Not Speculate Beyond the Official API

- Do not add support for undocumented fields, guessed response formats, or speculative compatibility behavior without a strong reason.
- Do not optimize for hypothetical providers or future shapes inside an existing adapter.
- If a provider exposes multiple formats, prefer the documented/default format over obscure alternatives unless there is a demonstrated need.

## 4. OpenAI-Compatible Requires Extra Discipline

- Treat OpenAI-compatible adapters as compatibility adapters, not as a place to accumulate arbitrary vendor quirks.
- Stay aligned with the canonical OpenAI wire contract by default.
- Only introduce vendor-specific deviations behind explicit configuration, clear naming, and tests.
- Do not let one compatible vendor's undocumented behavior redefine the default behavior for all compatible endpoints.

## 5. Preserve Meaning Across the Boundary

When changing adapter behavior, think in terms of contract preservation:

- Outbound: does the request still mean the same thing to the provider?
- Inbound: does the normalized response still mean the same thing to the caller?
- Errors: does the mapped error preserve the operational meaning of the upstream failure?
- Streaming: does event ordering, completion, and partial-data handling still match the provider's actual contract?

## 6. Be Conservative with Provider-Specific Extensions

- If provider metadata must be preserved, prefer carrying it through structured metadata fields rather than flattening or reinterpreting it.
- Keep provider-specific passthrough behavior explicit and narrow.
- Avoid convenience transformations that make debugging the original provider payload harder.

## 7. Favor Robustness at Boundaries

Adapters should be strict enough to catch invalid or malformed upstream data, but tolerant enough to handle realistic provider variation where the official API allows it.

- Validate required structural assumptions.
- Handle partial or boundary stream conditions carefully.
- Distinguish clearly between invalid upstream data and valid-but-empty content.
- Do not swallow provider errors or malformed payloads just to keep the happy path moving.

## 8. Tests Must Cover Contract Risk

When adapter behavior changes, tests should focus on contract risk rather than test volume.

Prefer coverage for:

- Outbound request serialization when request shape is part of correctness.
- Inbound normalization for public response fields.
- Error mapping and retry classification where relevant.
- Streaming boundaries, partial frames, EOF behavior, and completion semantics.
- Provider-specific features that have no cross-provider fallback.

One precise regression test is better than many shallow adapter tests.

## 9. Write Durable Guidance, Not Local API Inventories

This file should remain principle-based.

- Do not turn it into an exhaustive list of every request field or response field used by each provider.
- Do not mirror large portions of upstream provider docs here.
- Add stable decision rules, not long maintenance-heavy checklists.

If detailed provider notes become necessary, keep them narrow, local, and clearly tied to a real compatibility risk.
