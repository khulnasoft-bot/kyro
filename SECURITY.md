# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |

## Reporting a Vulnerability

Open a private security advisory on GitHub rather than a public issue.
Operational safety notes: the Docker image runs as a non-root user, input
sizes are capped (`KYRO_MAX_*` limits), and dependencies are audited in CI
via `cargo audit`. Do not expose `/v1/chat/completions` without a reverse
proxy enforcing auth/rate limits in production.
