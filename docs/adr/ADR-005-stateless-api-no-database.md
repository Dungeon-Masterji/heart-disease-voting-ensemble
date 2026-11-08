# ADR-005: Stateless API, in-memory rate limiter, no database

**Status:** Accepted (Week 9)

## Decision
`request -> validate -> preprocess -> predict -> respond`; no persistence. The
model artifact is checksum-verified and loaded once at startup. Rate limiting is
an in-process sliding window (no extra dependency).

## Consequences
- Simple to run and test; horizontally scalable for scoring.
- The rate limiter is per-process: with several workers the effective limit is
  multiplied. A shared store (e.g. Redis) would be needed for production.
- No audit trail of predictions. Logs deliberately exclude request bodies and
  predictions (no PHI); an audit table would be added only if governance needs it.
- The model is a pickle: the SHA-256 check detects tampering/corruption but
  pickles must still only come from a trusted build pipeline.
