"""
utils/media_detection.py – Advanced Road Hazard Detection Engine v3
=======================================================================
Key fixes in v3:
  - Mask BOTH white AND yellow lane markings before edge / crack scoring
  - Wider side-cut (15 %) to suppress guardrail / vegetation edges
  - Pothole contrast gate raised to 35 grey-levels
  - Minimum pothole blob area raised to 1 200 px
  - Edge-density damage threshold raised from 0.015 → 0.030
  - Texture-variance damage threshold raised from 100 → 180
  - Crack pixel-ratio threshold raised from 0.08 → 0.14
  - Surface-roughness threshold raised from 18 → 26
  - Dark-ratio threshold raised from 0.08 → 0.12
  - Good-road score ceiling stays < 15; damaged road scores ≥ 35
"""

import cv2
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

# ── Label registries ───────────────────────────────────────────────────────────
DAMAGE_LABELS = {
    "Pothole":       {"color": (0, 128, 255),  "severity_weight": 4},
    "Crack":         {"color": (0, 215, 255),  "severity_weight": 3},
    "Rutting":       {"color": (0, 255, 128),  "severity_weight": 3},
    "Edge Break":    {"color": (180, 0, 255),  "severity_weight": 3},
    "Debris":        {"color": (255, 165, 0),  "severity_weight": 2},
    "Water Logging": {"color": (255, 80, 80),  "severity_weight": 4},
    "Road Collapse": {"color": (0, 0, 220),    "severity_weight": 5},
}

ACCIDENT_LABELS = {
    "Accident":     {"color": (0, 0, 255),   "severity_weight": 5},
    "Collision":    {"color": (0, 0, 200),   "severity_weight": 5},
    "Vehicle Skid": {"color": (30, 30, 255), "severity_weight": 3},
    "Rollover":     {"color": (0, 0, 180),   "severity_weight": 5},
}


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    color: Tuple[int, int, int]
    severity: str
    category: str


@dataclass
class FrameResult:
    frame_index: int
    timestamp_sec: float
    detections: List[Detection] = field(default_factory=list)
    road_condition: str = "Good"
    damage_score: float = 0.0
    accident_detected: bool = False


# ── Severity / condition mappers ───────────────────────────────────────────────
def _score_to_severity(score: float) -> str:
    if score < 20:  return "Low"
    if score < 45:  return "Moderate"
    if score < 65:  return "High"
    if score < 80:  return "Severe"
    return "Critical"


def _score_to_condition(score: float) -> str:
    if score < 15:  return "Good"
    if score < 35:  return "Fair"
    if score < 55:  return "Poor"
    if score < 75:  return "Very Poor"
    return "Critical"


def _damage_score_from_detections(detections: List[Detection]) -> float:
    if not detections:
        return 0.0
    total = sum(
        DAMAGE_LABELS.get(d.label, {"severity_weight": 1})["severity_weight"]
        * d.confidence * 18
        for d in detections
    )
    return min(100.0, total)


# ── Build a combined lane-marking mask (white + yellow) ───────────────────────
def _lane_marking_mask(gray: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask where lane markings (white AND yellow) = 255.
    Dilated so nearby edges are also suppressed.
    """
    # White lane markings (high brightness)
    _, white = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

    # Yellow lane markings in HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 60, 100), (40, 255, 255))

    lane = cv2.bitwise_or(white, yellow)
    # Dilate to cover the edges right next to lane markings
    kernel = np.ones((7, 31), np.uint8)
    lane_dilated = cv2.dilate(lane, kernel)
    return lane_dilated


# ── Road mask: isolate only the road surface region ───────────────────────────
def _get_road_mask(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    h, w = image_bgr.shape[:2]
    # Bottom 45 % of frame – avoids sky, horizon, distant trees
    y_start = int(h * 0.55)
    road_roi = image_bgr[y_start:, :].copy()

    # Cut 20 % sides – suppresses guardrails, vegetation, buildings
    side_cut = int(w * 0.20)
    road_roi[:, :side_cut] = 0
    road_roi[:, w - side_cut:] = 0

    gray_roi = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
    return road_roi, gray_roi, y_start


# ── Sky/vegetation mask ────────────────────────────────────────────────────────
def _mask_vegetation_and_sky(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    veg_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    sky_mask  = cv2.inRange(hsv, (90, 50, 100), (130, 255, 255))
    combined  = cv2.bitwise_or(veg_mask, sky_mask)
    return cv2.bitwise_not(combined)


# ── Feature extraction (road ROI only) ────────────────────────────────────────
def _analyze_road_features(road_roi: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
    h, w = gray.shape

    # ── Lane mask (white + yellow markings) ──────────────────────────────────
    lane_mask = _lane_marking_mask(gray, road_roi)

    # ── 0. Road surface uniformity (CoV) — KEY good-road indicator ───────────
    # Exclude lane markings and zeroed-out side borders from uniformity check
    road_only_mask = cv2.bitwise_not(lane_mask)
    road_pixels = gray[road_only_mask > 0]
    road_mean = float(np.mean(road_pixels)) if len(road_pixels) > 0 else 128.0
    road_std  = float(np.std(road_pixels))  if len(road_pixels) > 0 else 0.0
    # Coefficient of Variation: smooth asphalt ≈ 0.05–0.15; damaged > 0.25
    road_cov  = road_std / (road_mean + 1e-6)

    # Lane marking fraction: well-marked roads score higher → good indicator
    lane_fraction = float((lane_mask > 0).sum()) / (h * w + 1e-6)

    # ── 1. Edge density (lane markings fully removed) ─────────────────────────
    median_brightness = float(np.median(gray))
    low_thresh  = max(25, median_brightness * 0.45)
    high_thresh = max(70, median_brightness * 0.95)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    edges_no_lanes = cv2.bitwise_and(edges, cv2.bitwise_not(lane_mask))
    edge_density = float(edges_no_lanes.sum()) / (h * w * 255 + 1e-6)

    # ── 2. Texture variance (Laplacian) ──────────────────────────────────────
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture_var = float(lap.var())

    # ── 3. Pothole blob detection ─────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (11, 11), 0)
    local_median = float(np.median(blurred))
    dark_thresh = max(30, local_median * 0.60)
    _, blob_bin = cv2.threshold(blurred, int(dark_thresh), 255, cv2.THRESH_BINARY_INV)
    blob_bin = cv2.bitwise_and(blob_bin, cv2.bitwise_not(lane_mask))

    contours, _ = cv2.findContours(blob_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = h * w * 0.15
    pothole_blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1500 or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / (bh + 1e-6)
        if not (0.3 <= aspect <= 3.5):
            continue
        mask_blob = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask_blob, [c], -1, 255, -1)
        surround_mask = cv2.dilate(mask_blob, np.ones((15, 15), np.uint8))
        surround_mask = cv2.bitwise_and(surround_mask, cv2.bitwise_not(mask_blob))
        blob_mean = float(cv2.mean(gray, mask=mask_blob)[0])
        surr_mean = float(cv2.mean(gray, mask=surround_mask)[0]) if surround_mask.sum() > 0 else blob_mean
        contrast  = surr_mean - blob_mean
        # Strong contrast gate: only genuine dark holes qualify
        if contrast > 40:
            pothole_blobs.append((c, area, contrast))

    # ── 4. Dark pixel ratio ───────────────────────────────────────────────────
    _, dark_mask = cv2.threshold(gray, int(local_median * 0.55), 255, cv2.THRESH_BINARY_INV)
    dark_ratio = float(dark_mask.sum()) / (h * w * 255 + 1e-6)

    # ── 5. Surface roughness (local block std) ────────────────────────────────
    block_size = 32
    local_stds = []
    for row in range(0, h - block_size, block_size):
        for col in range(0, w - block_size, block_size):
            patch = gray[row:row+block_size, col:col+block_size]
            local_stds.append(float(patch.std()))
    surface_roughness = float(np.mean(local_stds)) if local_stds else 0.0

    # ── 6. Crack detection (Sobel, lane-marking suppressed) ───────────────────
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    lane_mask_bool  = (lane_mask > 0)
    crack_pixels    = (sobel_mag > 70) & ~lane_mask_bool
    crack_pixel_ratio = float(crack_pixels.sum()) / (h * w + 1e-6)

    return {
        "edge_density":      edge_density,
        "texture_var":       texture_var,
        "pothole_blobs":     pothole_blobs,
        "pothole_count":     len(pothole_blobs),
        "dark_ratio":        dark_ratio,
        "surface_roughness": surface_roughness,
        "crack_pixel_ratio": crack_pixel_ratio,
        "local_median":      local_median,
        "road_cov":          road_cov,
        "lane_fraction":     lane_fraction,
    }


# ── Convert features to damage score ──────────────────────────────────────────
def _features_to_damage_score(feat: Dict[str, Any]) -> float:
    """
    Calibrated so smooth asphalt scores < 12 (Good).
    Damaged road (real potholes/cracks) scores >= 35.

    SMOOTHNESS OVERRIDE: if road CoV < 0.18 AND no validated potholes,
    the road is smooth asphalt → clamp score to max 10 (always Good).
    """
    cov           = feat.get("road_cov", 0.5)
    lane_fraction = feat.get("lane_fraction", 0.0)

    # ── Smoothness fast-path ─────────────────────────────────────────────────
    # A well-maintained road: low CoV (uniform surface) + visible lane markings
    # + zero validated potholes → always Good, regardless of other metrics.
    if cov < 0.18 and feat["pothole_count"] == 0:
        return 0.0   # ← definitive GOOD ROAD, no further scoring needed

    # ── If CoV is moderate but no potholes, cap at Fair (score ≤ 30) ─────────
    if cov < 0.25 and feat["pothole_count"] == 0:
        score_cap = 30.0
    else:
        score_cap = 100.0

    score = 0.0

    # Edge density (lane markings removed)
    ed = feat["edge_density"]
    if ed > 0.035:
        score += min(28, (ed - 0.035) * 550)

    # Texture variance
    tv = feat["texture_var"]
    if tv > 200:
        score += min(20, (tv - 200) / 55)

    # Validated pothole blobs (shape + high contrast)
    score += min(35, feat["pothole_count"] * 12)

    # Dark ratio
    dr = feat["dark_ratio"]
    if dr > 0.14:
        score += min(15, (dr - 0.14) * 120)

    # Surface roughness
    sr = feat["surface_roughness"]
    if sr > 28:
        score += min(12, (sr - 28) * 0.9)

    # Crack pixel ratio
    cp = feat["crack_pixel_ratio"]
    if cp > 0.15:
        score += min(10, (cp - 0.15) * 170)

    return round(min(score_cap, score), 1)


# ── Build detections from validated features ───────────────────────────────────
def _build_detections(
    image_bgr: np.ndarray,
    feat: Dict[str, Any],
    score: float,
    y_offset: int,
    gray_road: np.ndarray,
    road_roi: np.ndarray,
) -> List[Detection]:
    detections: List[Detection] = []
    h_full, w_full = image_bgr.shape[:2]

    # ── Potholes ──────────────────────────────────────────────────────────────
    if score >= 22 and feat["pothole_count"] > 0:
        for contour, area, contrast in feat["pothole_blobs"][:5]:
            x, y, bw, bh = cv2.boundingRect(contour)
            pad = 10
            x  = max(0, x - pad)
            y  = max(0, y - pad)
            bw = min(w_full - x, bw + pad * 2)
            bh = min(gray_road.shape[0] - y, bh + pad * 2)
            conf = round(min(0.96, 0.50 + (area / (gray_road.shape[0] * w_full)) * 5 + contrast / 250), 2)
            detections.append(Detection(
                label="Pothole",
                confidence=conf,
                bbox=(x, y + y_offset, bw, bh),
                color=DAMAGE_LABELS["Pothole"]["color"],
                severity=_score_to_severity(score),
                category="damage",
            ))

    # ── Cracks ────────────────────────────────────────────────────────────────
    if score >= 35 and feat["edge_density"] > 0.035:
        lane_mask = _lane_marking_mask(gray_road, road_roi)
        blurred   = cv2.GaussianBlur(gray_road, (5, 5), 0)
        edges     = cv2.Canny(blurred, 40, 110)
        edges     = cv2.bitwise_and(edges, cv2.bitwise_not(lane_mask))
        kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated   = cv2.dilate(edges, kernel)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        added = 0
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(c) < 600 or added >= 2:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 40:
                continue
            conf = round(min(0.90, 0.48 + feat["edge_density"] * 5), 2)
            detections.append(Detection(
                label="Crack",
                confidence=conf,
                bbox=(x, y + y_offset, bw, bh),
                color=DAMAGE_LABELS["Crack"]["color"],
                severity=_score_to_severity(score * 0.8),
                category="damage",
            ))
            added += 1

    # ── Water Logging ─────────────────────────────────────────────────────────
    if score >= 40 and feat["dark_ratio"] > 0.15:
        _, wl = cv2.threshold(gray_road, int(feat["local_median"] * 0.50), 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(wl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours[:1]:
            if cv2.contourArea(c) < 3000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            conf = round(min(0.88, 0.55 + feat["dark_ratio"] * 2), 2)
            detections.append(Detection(
                label="Water Logging",
                confidence=conf,
                bbox=(x, y + y_offset, bw, bh),
                color=DAMAGE_LABELS["Water Logging"]["color"],
                severity="High",
                category="damage",
            ))

    return detections


# ── Main image analysis entry point ───────────────────────────────────────────
def mock_analyze_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Good roads score < 15  → condition: Good, no detections.
    Damaged roads score ≥ 35 → condition: Poor / Worse.
    """
    h, w = image_bgr.shape[:2]

    road_roi, gray_road, y_offset = _get_road_mask(image_bgr.copy())
    feat  = _analyze_road_features(road_roi, gray_road)
    score = _features_to_damage_score(feat)

    accident_detected = False
    detections: List[Detection] = []

    if score > 72 and random.random() < 0.10:
        label = random.choice(list(ACCIDENT_LABELS.keys()))
        info  = ACCIDENT_LABELS[label]
        bx = w // 6; by = h // 6
        bw = w // 2; bh = h // 2
        conf = round(min(0.95, 0.68 + (score / 100) * 0.25), 2)
        detections.append(Detection(
            label=label, confidence=conf,
            bbox=(bx, by, bw, bh), color=info["color"],
            severity="Severe", category="accident",
        ))
        accident_detected = True
        score = max(score, 72.0)
    else:
        detections = _build_detections(image_bgr, feat, score, y_offset, gray_road, road_roi)

    condition = _score_to_condition(score)
    severity  = _score_to_severity(score)
    annotated = _draw_on_image(image_bgr.copy(), detections, score, condition)

    return {
        "detections":        detections,
        "damage_score":      round(score, 1),
        "road_condition":    condition,
        "severity":          severity,
        "accident_detected": accident_detected,
        "annotated_image":   annotated,
        "summary":           _build_image_summary(detections, score, condition),
        "features":          {k: v for k, v in feat.items() if k != "pothole_blobs"},
    }


# ── Video frame analysis ───────────────────────────────────────────────────────
def mock_analyze_video_frame(frame_bgr: np.ndarray,
                              frame_idx: int,
                              fps: float) -> FrameResult:
    road_roi, gray_road, y_offset = _get_road_mask(frame_bgr.copy())
    feat  = _analyze_road_features(road_roi, gray_road)
    score = _features_to_damage_score(feat)
    score = min(100.0, max(0.0, score + random.uniform(-2, 2)))

    dets: List[Detection] = []
    acc = False

    if score > 70 and random.random() < 0.05:
        h, w  = frame_bgr.shape[:2]
        label = random.choice(list(ACCIDENT_LABELS.keys()))
        info  = ACCIDENT_LABELS[label]
        dets.append(Detection(
            label=label,
            confidence=round(min(0.95, 0.70 + (score / 100) * 0.23), 2),
            bbox=(w // 6, h // 6, w // 2, h // 2),
            color=info["color"],
            severity="Severe",
            category="accident",
        ))
        acc   = True
        score = max(score, 70.0)
    else:
        dets = _build_detections(frame_bgr, feat, score, y_offset, gray_road, road_roi)

    return FrameResult(
        frame_index=frame_idx,
        timestamp_sec=round(frame_idx / max(fps, 1), 2),
        detections=dets,
        road_condition=_score_to_condition(score),
        damage_score=round(score, 1),
        accident_detected=acc,
    )


# ── Annotation helpers ─────────────────────────────────────────────────────────
def _draw_on_image(img: np.ndarray,
                   detections: List[Detection],
                   score: float,
                   condition: str) -> np.ndarray:
    overlay = img.copy()

    for d in detections:
        x, y, bw, bh = d.bbox
        x, y = max(0, x), max(0, y)
        bw   = min(bw, img.shape[1] - x)
        bh   = min(bh, img.shape[0] - y)
        if bw <= 0 or bh <= 0:
            continue
        c = d.color
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), c, 2)
        lbl = f"{d.label} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cy = max(y - 4, th + 6)
        cv2.rectangle(overlay, (x, cy - th - 6), (x + tw + 8, cy + 2), c, -1)
        cv2.putText(overlay, lbl, (x + 4, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    # HUD bar
    hud_h = 50
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], hud_h), (14, 17, 23), -1)
    cond_bgr = {
        "Good":      (80, 200, 80),
        "Fair":      (80, 210, 180),
        "Poor":      (0, 170, 255),
        "Very Poor": (0, 80, 255),
        "Critical":  (0, 0, 220),
    }.get(condition, (200, 200, 200))

    cv2.putText(overlay, "VigiRoad AI  |  Road Condition Analysis",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(overlay, f"Condition: {condition}   Damage Score: {score:.0f}/100",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cond_bgr, 1)

    bar_w = int((score / 100) * (overlay.shape[1] - 20))
    cv2.rectangle(overlay, (10, hud_h + 3), (10 + bar_w, hud_h + 9), cond_bgr, -1)
    cv2.rectangle(overlay, (10, hud_h + 3), (overlay.shape[1] - 10, hud_h + 9), (60, 60, 60), 1)

    return overlay


def annotate_video_frame(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    return _draw_on_image(
        frame.copy(), result.detections,
        result.damage_score, result.road_condition,
    )


def _build_image_summary(detections, score, condition) -> List[str]:
    lines = []
    if not detections:
        if score < 15:
            lines.append("✅ Road surface is in good condition. No damage detected.")
        else:
            lines.append(f"⚠️ Road surface shows signs of wear (score {score:.0f}/100). "
                         "No specific damage regions localised.")
    else:
        dam = [d for d in detections if d.category == "damage"]
        acc = [d for d in detections if d.category == "accident"]
        if acc:
            lines.append(f"🚨 ACCIDENT DETECTED: {', '.join(set(d.label for d in acc))}")
        if dam:
            counts: Dict[str, int] = {}
            for d in dam:
                counts[d.label] = counts.get(d.label, 0) + 1
            for label, cnt in counts.items():
                lines.append(f"🕳️ {label}: {cnt} instance(s)")
    lines.append(f"📊 Damage Score: {score:.0f}/100  —  Condition: {condition}")
    return lines


def build_video_summary(frame_results: List[FrameResult], fps: float) -> Dict[str, Any]:
    n               = len(frame_results)
    damaged_frames  = [f for f in frame_results
                       if f.detections and any(d.category == "damage" for d in f.detections)]
    accident_frames = [f for f in frame_results if f.accident_detected]
    avg_score       = sum(f.damage_score for f in frame_results) / n if n else 0
    peak_score      = max((f.damage_score for f in frame_results), default=0)
    damage_types: Dict[str, int] = {}
    for fr in frame_results:
        for d in fr.detections:
            damage_types[d.label] = damage_types.get(d.label, 0) + 1

    return {
        "total_frames":         n,
        "damaged_frame_count":  len(damaged_frames),
        "accident_frame_count": len(accident_frames),
        "accident_timestamps":  [f.timestamp_sec for f in accident_frames],
        "avg_damage_score":     round(avg_score, 1),
        "peak_damage_score":    round(peak_score, 1),
        "overall_condition":    _score_to_condition(avg_score),
        "peak_condition":       _score_to_condition(peak_score),
        "damage_type_counts":   damage_types,
        "severity":             _score_to_severity(avg_score),
    }
