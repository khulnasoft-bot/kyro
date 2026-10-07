# --- Stage 1: Builder ---
FROM rustlang/rust:nightly-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/kyro

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY examples/Cargo.toml examples/Cargo.toml

# Create dummy source to pre-build dependencies (caching)
RUN mkdir src examples/src benches && echo "fn main() {}" > src/main.rs \
    && for f in pattern_matching tokenization attention embedding kv_demo small_trainer; do \
         echo "fn main() {}" > examples/src/$f.rs; done \
    && echo "fn main() {}" > benches/scheduler_bench.rs
RUN cargo build --release

# Copy actual source
COPY . .

# Build the real binary with optimizations
RUN cargo build --release

# --- Stage 2: Runtime ---
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Run as an unprivileged user with no shell for minimal privilege
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin kyro
USER kyro

# Copy binary from builder
COPY --from=builder /usr/src/kyro/target/release/kyro /usr/local/bin/kyro

# Expose API port
EXPOSE 3000

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Run the engine
ENTRYPOINT ["kyro"]
