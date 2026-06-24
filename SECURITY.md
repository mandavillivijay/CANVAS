# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities by emailing **mvijayfromvizag@gmail.com** with the subject line `[CANVAS] Security Vulnerability`.

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- Any suggested mitigations you are aware of

### Response SLA

| Milestone | Target |
|-----------|--------|
| Acknowledgement | Within 48 hours |
| Initial assessment | Within 7 days |
| Patch / mitigation | Within 30 days for critical, 90 days for others |

We follow responsible disclosure: once a fix is available we will coordinate a public disclosure date with you. We ask that you refrain from publishing details until a patch is released.

## Scope

In scope: `canvas_heal` Python package, CLI, pytest plugin, CI workflows in this repository.

Out of scope: third-party models downloaded from HuggingFace Hub, upstream `sentence-transformers` or `playwright` vulnerabilities (report those upstream).
