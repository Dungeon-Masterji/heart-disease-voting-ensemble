# ADR-004: Serve soft voting (probabilities), regardless of which strategy scored higher

**Status:** Accepted (Week 7)

## Context
Hard voting returns only a label, so it cannot support a threshold or a
probability display. In nested CV the random search chose hard voting in
several folds, leaving ROC AUC undefined and Week 8 without probabilities.

## Decision
Restrict the final search to `voting="soft"`. The reason is a product
requirement (threshold + probability), not a performance win.

## Consequences
- Tuned soft voting matched logistic regression within fold noise on 302
  patients; the model card reports this plainly instead of claiming an
  improvement.
