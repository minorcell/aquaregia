FROM rust:1.92-bookworm AS builder

WORKDIR /src
COPY . .
RUN cargo build --release --manifest-path examples/remote_agent/Cargo.toml --bin sandbox_mcp

FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY --from=builder /src/examples/remote_agent/target/release/sandbox_mcp /usr/local/bin/sandbox_mcp

ENV SANDBOX_MCP_ADDR=0.0.0.0:8931
ENV SANDBOX_WORKDIR=/workspace
EXPOSE 8931

CMD ["sandbox_mcp"]
