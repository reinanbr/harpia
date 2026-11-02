# Security Policy

## Supported versions

Only the latest released minor version receives security fixes before 1.0.

## Reporting a vulnerability

Please **do not** open a public issue. Use GitHub's private vulnerability reporting
("Security" tab → "Report a vulnerability") on the Harpia repository. Include a description,
affected versions and, if possible, a minimal reproduction. We aim to acknowledge reports within
72 hours and to publish a fix and advisory once users can upgrade.

## Scope notes

- Harpia never opens network connections unless the caller passes `NetworkPolicy::Online`; a
  connection made under `Offline` is a security bug.
- Secrets (API keys, tokens) must never appear in logs, errors or `Debug` output.
- Parsers (`harpia-structured`, dataset readers, chat templates) process untrusted input and
  must not panic or allocate unboundedly on malicious data; such failures are in scope.
- Chat templates are rendered in a sandboxed `minijinja` environment without filesystem or
  network access.
