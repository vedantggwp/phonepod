# Security Policy

## Supported Versions

phonepod is pre-1.0 software. Security fixes target the `main` branch and the latest public beta release.

## Reporting a Vulnerability

Please do not open a public issue for vulnerabilities involving unsafe file handling, dependency compromise, private audio leakage, or unexpected network access.

Report privately by opening a GitHub security advisory for this repository if available, or contact the maintainer through the email on the GitHub profile.

Useful reports include:

- affected command, module, or file format;
- reproduction steps using non-sensitive sample audio;
- whether the issue involves local files, temporary files, model downloads, or the optional web UI;
- expected and observed behavior.

## Privacy Boundary

phonepod is designed as local-first audio tooling. Public issues and PRs should not include private recordings, transcripts, credentials, model cache contents, or proprietary audio samples.
