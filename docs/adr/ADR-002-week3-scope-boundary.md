# ADR-002: Hold the repository strictly to Week 1-3 PRD scope

**Status:** Accepted (Week 3)

## Context
The 12-week PRD roadmap (§17) scopes Weeks 1-3 to research, requirements,
and data ingestion/validation/EDA only. The Voting Classifier, FastAPI
service, and React UI are Weeks 5-11 deliverables.

## Decision
This repository currently ships only what Weeks 1-3 deliver:
data ingestion, validation, and EDA, with real generated artifacts and
passing tests. `backend/` and `frontend/` are intentionally not present yet.
The README states this scope explicitly instead of describing the full
12-week end state as if it were already built.

## Consequences
- No FastAPI, no React, no trained model, no Voting Classifier code exist in
  this snapshot — they will be added in the corresponding weeks per the
  PRD roadmap, each with their own working code and tests before being
  described in the README.
- Badges, API examples, and setup instructions for unbuilt components are
  removed from the README rather than presented as if functional.
