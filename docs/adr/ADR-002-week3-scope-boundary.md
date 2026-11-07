# ADR-002: Preserve Week 1-3 scope boundary

**Status:** Superseded; retained as a historical Week 3 decision.

## Context
The 12-week PRD roadmap (§17) scopes Weeks 1-3 to research, requirements,
and data ingestion/validation/EDA only. The Voting Classifier, FastAPI
service, and React UI are Weeks 5-11 deliverables.

## Decision
At its original Week 3 decision point, this repository shipped only data
ingestion, validation, and EDA, with real generated artifacts and passing
tests. Later project chunks added the training, API, frontend, Docker, and
documentation work described by the full PRD. This ADR is preserved as
historical context, not as a current repository-state statement.

## Consequences
- The original decision prevented later deliverables from being implied
  before they had working code and tests.
- Badges, API examples, and setup instructions for unbuilt components are
  removed from the README rather than presented as if functional.
