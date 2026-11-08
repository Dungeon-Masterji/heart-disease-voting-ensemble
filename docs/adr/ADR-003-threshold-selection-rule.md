# ADR-003: Choose the operating threshold by a sensitivity-first rule

**Status:** Accepted (Week 8)

## Context
The default 0.5 cut-off treats both error types equally. In screening, a missed
case (false negative) is costlier than an unnecessary follow-up (false positive).

## Decision
Pick the highest-specificity threshold whose out-of-fold sensitivity is at least
0.90 (`TARGET_SENS` in the notebook). Probabilities come from nested-CV outer
folds, so every patient was scored by a model that never saw them. A second,
"honest" check re-selects the threshold inside each fold using only the other
four folds and reports sensitivity/specificity on the held-out fold.

## Consequences
- The 0.90 target is a documented judgement, not a law; a clinical stakeholder
  could justify a different target.
- Selecting and scoring on the same predictions is slightly optimistic; the
  honest check quantifies this and is the number reported as the main result.
- Dataset prevalence (~54%) is far higher than real screening, so real-world
  precision would be much lower. Stated in the model card.
