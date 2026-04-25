# Experiments

## Goal

The goal of this challenge is to determine whether each prior patient examination should be shown to a radiologist while reading the current examination. The API receives one or more cases, and each case contains one current study plus multiple prior studies for the same patient. The output must include exactly one prediction for every prior study.

## Baseline

The first baseline was a simple modality-matching rule:

- If the current and prior study had the same modality, predict relevant.
- Otherwise, predict not relevant.

This was easy to implement, fast, and guaranteed that every prior study received a prediction. However, it was too broad because exams with the same modality but different anatomy can be unrelated. For example, an MRI brain and MRI knee should not usually be treated as relevant to each other.

## Improved Rule-Based Approach

The final endpoint uses a deterministic scoring system based on:

- modality overlap, such as CT with CT or MRI with MRI
- anatomy/body-part overlap, such as brain, chest, abdomen, pelvis, spine, breast, or extremity
- normalized keyword similarity between study descriptions
- contrast similarity when available
- recency of the prior study

Anatomy/body-part overlap is weighted strongly because prior examinations are usually most useful when they involve the same clinical region as the current study. Modality agreement adds confidence, but cross-modality priors can still be relevant when the anatomy matches, such as CT head and MRI brain.

## What Worked

The strongest improvement came from separating modality matching from anatomy matching. Same-modality-only logic was too noisy, while anatomy-aware matching captured more clinically meaningful relationships.

The API is also designed to process all cases and priors in a single request without external calls. This keeps latency low and avoids timeout risk during hidden evaluation.

## What Failed

A pure exact-string match was too strict because study descriptions can use different wording for similar exams. For example:

- CT HEAD WITHOUT CNTRST
- CT HEAD WITHOUT CONTRAST
- MRI BRAIN STROKE LIMITED WITHOUT CONTRAST

A pure modality match was too loose because different body regions can share the same modality.

## Current Limitations

The approach is rule-based and does not learn from the public labeled split. It may miss subtle clinical relevance patterns, such as oncology follow-up studies or related vascular exams where anatomy is implied indirectly.

The keyword dictionaries are also manually defined, so performance depends on whether the study description contains recognizable terms.

## Next Improvements

With more time, I would improve the solution by:

1. Training a binary classifier using the public labeled split.
2. Adding TF-IDF or sentence-embedding similarity between current and prior study descriptions.
3. Learning optimal thresholds from validation data.
4. Expanding clinical anatomy normalization using radiology-specific terminology.
5. Adding error analysis for false positives and false negatives by modality and body region.
6. Caching repeated current-prior pairs for faster repeated evaluation.

## Deployment Notes

The submitted API exposes a POST `/predict` endpoint. It returns a JSON object with a `predictions` list. Each prediction contains:

- `case_id`
- `study_id`
- `predicted_is_relevant`

The endpoint does not call any external service, which avoids API latency and private-evaluation timeout risks.
