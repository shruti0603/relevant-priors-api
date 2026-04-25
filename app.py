from flask import Flask, request, jsonify
from datetime import datetime
import re
import math
import logging
from typing import Dict, List, Any, Tuple

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

MODALITIES = {
    "MRI": ["mri", "mr "],
    "CT": ["ct ", "computed tomography"],
    "XR": ["xray", "x-ray", "xr ", "radiograph"],
    "US": ["ultrasound", "us "],
    "MG": ["mammogram", "mammography"],
    "NM": ["nuclear medicine", "pet", "spect"],
    "FL": ["fluoro", "fluoroscopy"],
}

BODY_PART_KEYWORDS = {
    "brain_head": ["brain", "head", "skull", "stroke", "intracranial", "orbits", "sinus", "maxillofacial", "facial"],
    "neck": ["neck", "cervical", "c-spine", "soft tissue neck"],
    "chest": ["chest", "thorax", "lung", "pulmonary", "ribs", "sternum"],
    "heart": ["cardiac", "heart", "coronary", "cta coronary"],
    "abdomen": ["abdomen", "abdominal", "liver", "gallbladder", "kidney", "renal", "pancreas", "spleen"],
    "pelvis": ["pelvis", "pelvic", "hip", "bladder", "uterus", "prostate"],
    "spine_cervical": ["cervical spine", "c-spine"],
    "spine_thoracic": ["thoracic spine", "t-spine"],
    "spine_lumbar": ["lumbar spine", "l-spine"],
    "spine": ["spine", "spinal"],
    "shoulder": ["shoulder", "clavicle"],
    "arm": ["humerus", "elbow", "forearm", "wrist", "hand", "finger"],
    "leg": ["femur", "knee", "tibia", "fibula", "ankle", "foot", "toe"],
    "breast": ["breast", "mammogram", "mammography"],
    "vascular": ["angiogram", "angiography", "cta", "mra", "vascular", "artery", "vein", "venous"],
}

CONTRAST_TERMS = {
    "with_contrast": ["with contrast", "w contrast", "w/ contrast"],
    "without_contrast": ["without contrast", "wo contrast", "w/o contrast", "non contrast", "no contrast"],
}

STOPWORDS = {
    "with", "without", "contrast", "cntrst", "limited", "complete", "routine",
    "portable", "single", "multiple", "view", "views", "exam", "study", "and", "or",
    "left", "right", "bilateral", "follow", "up", "for", "of", "the", "w", "wo"
}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("cntrst", "contrast")
    text = text.replace("w/o", "without")
    text = text.replace("wo ", "without ")
    text = text.replace("w/", "with")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_modality(desc: str) -> str:
    d = " " + normalize_text(desc) + " "
    for modality, keys in MODALITIES.items():
        for key in keys:
            if key in d:
                return modality
    first = normalize_text(desc).split()
    if first:
        token = first[0].upper()
        if token in MODALITIES:
            return token
    return "UNKNOWN"


def detect_body_parts(desc: str) -> set:
    d = normalize_text(desc)
    parts = set()
    for part, keys in BODY_PART_KEYWORDS.items():
        for key in keys:
            if key in d:
                parts.add(part)
                break

    # map specific spine regions to broader spine too
    if any(p.startswith("spine_") for p in parts):
        parts.add("spine")

    return parts


def detect_contrast(desc: str) -> str:
    d = normalize_text(desc)
    for label, keys in CONTRAST_TERMS.items():
        for key in keys:
            if key in d:
                return label
    return "unknown"


def tokenize(desc: str) -> set:
    d = normalize_text(desc)
    tokens = set(t for t in d.split() if len(t) > 1 and t not in STOPWORDS)
    return tokens


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def parse_date(date_str: str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


def years_between(current_date: str, prior_date: str) -> float:
    c = parse_date(current_date)
    p = parse_date(prior_date)
    if not c or not p:
        return 99.0
    return abs((c - p).days) / 365.25


def relevance_score(current: Dict[str, Any], prior: Dict[str, Any]) -> float:
    curr_desc = current.get("study_description", "")
    prior_desc = prior.get("study_description", "")

    curr_modality = detect_modality(curr_desc)
    prior_modality = detect_modality(prior_desc)

    curr_parts = detect_body_parts(curr_desc)
    prior_parts = detect_body_parts(prior_desc)

    curr_tokens = tokenize(curr_desc)
    prior_tokens = tokenize(prior_desc)

    curr_contrast = detect_contrast(curr_desc)
    prior_contrast = detect_contrast(prior_desc)

    age_years = years_between(
        current.get("study_date", ""),
        prior.get("study_date", "")
    )

    score = 0.0

    # Modality matters, but cross-modality priors can still be clinically useful.
    if curr_modality != "UNKNOWN" and curr_modality == prior_modality:
        score += 0.35
    elif curr_modality != "UNKNOWN" and prior_modality != "UNKNOWN":
        score -= 0.10

    # Same anatomy/body region is the strongest signal.
    part_overlap = curr_parts & prior_parts
    if part_overlap:
        score += 0.45
        if "spine" in part_overlap and len(part_overlap) == 1:
            score += 0.05
    else:
        # Vascular exams often relate to anatomy through CTA/MRA words.
        if "vascular" in curr_parts and "vascular" in prior_parts:
            score += 0.25
        else:
            score -= 0.20

    # Text similarity catches exact/similar descriptions.
    token_sim = jaccard(curr_tokens, prior_tokens)
    score += 0.30 * token_sim

    # Contrast difference should not kill relevance, but exact contrast helps slightly.
    if curr_contrast != "unknown" and curr_contrast == prior_contrast:
        score += 0.05

    # Recency helps, but old priors with same anatomy are still often relevant.
    if age_years <= 1:
        score += 0.12
    elif age_years <= 3:
        score += 0.08
    elif age_years <= 7:
        score += 0.03
    else:
        score -= 0.03

    # Strong exact-body rules.
    nd_curr = normalize_text(curr_desc)
    nd_prior = normalize_text(prior_desc)

    if nd_curr == nd_prior:
        score += 0.25

    # Brain CT/MRI prior is often useful for current brain/head exams.
    if ("brain_head" in curr_parts and "brain_head" in prior_parts):
        score += 0.10

    # Chest xray/CT priors are commonly relevant to chest studies.
    if ("chest" in curr_parts and "chest" in prior_parts):
        score += 0.10

    return score


def is_relevant(current: Dict[str, Any], prior: Dict[str, Any]) -> bool:
    score = relevance_score(current, prior)

    curr_desc = current.get("study_description", "")
    prior_desc = prior.get("study_description", "")

    curr_parts = detect_body_parts(curr_desc)
    prior_parts = detect_body_parts(prior_desc)
    curr_modality = detect_modality(curr_desc)
    prior_modality = detect_modality(prior_desc)

    # High-confidence positive rule.
    if curr_parts & prior_parts and curr_modality == prior_modality:
        return True

    # Same anatomy cross-modality can be relevant if score is decent.
    if curr_parts & prior_parts and score >= 0.35:
        return True

    # Otherwise threshold.
    return score >= 0.55


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Relevant priors API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        cases = data.get("cases", [])
        predictions = []

        total_priors = 0

        for case in cases:
            case_id = case.get("case_id")
            current = case.get("current_study", {})
            priors = case.get("prior_studies", [])

            total_priors += len(priors)

            for prior in priors:
                study_id = prior.get("study_id")
                predictions.append({
                    "case_id": str(case_id),
                    "study_id": str(study_id),
                    "predicted_is_relevant": bool(is_relevant(current, prior))
                })

        app.logger.info(
            "Processed request: challenge_id=%s, cases=%d, priors=%d, predictions=%d",
            data.get("challenge_id"),
            len(cases),
            total_priors,
            len(predictions),
        )

        return jsonify({"predictions": predictions})

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "prediction_failed",
            "message": str(e),
            "predictions": []
        }), 400


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
