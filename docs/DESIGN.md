# DESIGN.md for PM2.5 STGNN Thailand

This document is the current architecture and design reference for the project.

## Goals

- Build a spatial-temporal graph neural network for PM2.5 forecasting.
- Support dataset ingestion from Air4Thai and OpenAQ.
- Keep model and preprocessing code modular and testable.
- Separate architecture decisions from implementation details.

## Structure

- `src/` contains the implementation code.
- `docs/` contains design, session notes, and architecture rationale.
- `.claude/` contains agent definitions and Claude settings.

## Decision notes

- Prefer explicit agent roles: architect, implementer, tester, reviewer.
- Store project memory in `CLAUDE.md`.
- Use a modern Python packaging config via `pyproject.toml`.
- Keep environment examples out of source control via `.env.example`.
