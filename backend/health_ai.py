"""
Health AI utilities (CPU-friendly, beginner-friendly).

This module adds an "AI-based" pet health detection flow using:
- OpenCV (if available) for fast image preprocessing and simple visual signals
- A Pillow+NumPy fallback if OpenCV is not installed
- A lightweight scoring model to output:
  - condition: Healthy / Sick (possible)
  - issue: skin problems / infection signs / weakness indicators
  - confidence: percentage

Important:
This is designed for student demos and fast local execution on CPU.
For real clinical accuracy, you would train/fine-tune on veterinary datasets.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

try:  # optional dependency
    import cv2  # type: ignore

    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None
    _CV2_AVAILABLE = False


@dataclass(frozen=True)
class HealthPrediction:
    condition: str  # "Healthy" or "Sick"
    issue: Optional[str]  # "skin problems" | "infection signs" | "weakness indicators" | None
    confidence: float  # 0..1
    signals: Dict[str, float]

    def to_user_text(self) -> str:
        pct = round(self.confidence * 100, 1)
        if self.condition.lower() == "healthy":
            return f"Healthy (confidence {pct}%)"
        if self.issue:
            return f"Sick / possible issue: {self.issue} (confidence {pct}%)"
        return f"Sick / possible issue detected (confidence {pct}%)"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _read_bgr(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found on server.")
    if _CV2_AVAILABLE:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unsupported image format or corrupted file.")
        return img

    # Pillow fallback: read RGB then convert to BGR ndarray for consistent downstream logic.
    try:
        rgb = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unsupported image format or corrupted file: {exc}") from exc
    arr = np.asarray(rgb, dtype=np.uint8)
    bgr = arr[:, :, ::-1].copy()
    return bgr


def _preprocess(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - img_small_bgr: resized for speed
      - hsv: HSV version
      - gray: grayscale version
    """
    if _CV2_AVAILABLE:
        img_small = cv2.resize(img_bgr, (320, 320), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        return img_small, hsv, gray

    # Pillow+NumPy fallback
    rgb = img_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb)
    pil = pil.resize((320, 320), resample=Image.BILINEAR)
    rgb_small = np.asarray(pil, dtype=np.uint8)

    # HSV using Pillow conversion (returns 0..255 per channel)
    hsv_pil = pil.convert("HSV")
    hsv = np.asarray(hsv_pil, dtype=np.uint8)
    # Match OpenCV hue scale 0..179 approximately (Pillow hue 0..255)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 0] = hsv[:, :, 0] * (179.0 / 255.0)
    hsv = hsv.astype(np.uint8)

    # Grayscale
    gray = np.asarray(pil.convert("L"), dtype=np.uint8)

    # Return BGR small for consistency (even if unused)
    bgr_small = rgb_small[:, :, ::-1].copy()
    return bgr_small, hsv, gray


def _conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Minimal 2D convolution for fallback image metrics.
    img: HxW float32
    kernel: khxkw float32
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            region = padded[y : y + kh, x : x + kw]
            out[y, x] = float(np.sum(region * kernel))
    return out


def _edge_density(gray_u8: np.ndarray) -> float:
    if _CV2_AVAILABLE:
        edges = cv2.Canny(gray_u8, 80, 160)
        return float(np.mean((edges > 0).astype(np.float32)))

    g = gray_u8.astype(np.float32) / 255.0
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = _conv2d(g, kx)
    gy = _conv2d(g, ky)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean((mag > 0.25).astype(np.float32)))


def _laplacian_variance(gray_u8: np.ndarray) -> float:
    if _CV2_AVAILABLE:
        return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())

    g = gray_u8.astype(np.float32) / 255.0
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = _conv2d(g, k)
    return float(np.var(lap))


def _compute_signals(hsv: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
    """
    Compute simple image signals (0..1-ish) that correlate with *possible* issues:
    - redness: high red hue dominance can indicate inflammation/skin irritation
    - yellowish: can indicate discharge/dirty coat (very rough)
    - texture: high edge density can indicate rough patches / lesions / wounds
    - blur: extreme blur reduces reliability (used to reduce confidence)
    - brightness: under/over exposure can reduce reliability
    """
    h = hsv[:, :, 0].astype(np.float32)  # 0..179
    s = hsv[:, :, 1].astype(np.float32)  # 0..255
    v = hsv[:, :, 2].astype(np.float32)  # 0..255

    # Redness: hue near 0 or near 179 with decent saturation.
    sat_mask = (s > 60).astype(np.float32)
    red_mask = (((h < 10) | (h > 170)).astype(np.float32)) * sat_mask
    redness = float(np.mean(red_mask))

    # Yellow-ish hues (approx 20..40) with saturation.
    yellow_mask = (((h >= 18) & (h <= 42)).astype(np.float32)) * sat_mask
    yellowish = float(np.mean(yellow_mask))

    # Texture / lesion proxy: edge density in grayscale.
    texture = _edge_density(gray)

    # Blur estimate: variance of Laplacian (normalized).
    lap_var = _laplacian_variance(gray)
    # Typical ranges vary wildly; map to 0..1 in a stable way.
    # Fallback Laplacian variance is on a different scale; normalize conservatively.
    denom = 300.0 if _CV2_AVAILABLE else 0.06
    blur_quality = float(np.clip(lap_var / denom, 0.0, 1.0))

    # Exposure quality: prefer mid-range brightness.
    mean_v = float(np.mean(v) / 255.0)
    exposure_quality = float(1.0 - min(abs(mean_v - 0.55) / 0.55, 1.0))

    return {
        "redness": redness,
        "yellowish": yellowish,
        "texture": texture,
        "blur_quality": blur_quality,
        "exposure_quality": exposure_quality,
    }


def predict_health_from_image(image_path: str) -> HealthPrediction:
    """
    Predict a *possible* health condition from a pet image.

    Output is intentionally conservative and fast:
    - If signals are low → Healthy
    - If signals are high → Sick (possible) + issue label
    """
    img_bgr = _read_bgr(image_path)
    _img_small, hsv, gray = _preprocess(img_bgr)
    signals = _compute_signals(hsv=hsv, gray=gray)

    redness = signals["redness"]
    yellowish = signals["yellowish"]
    texture = signals["texture"]
    blur_q = signals["blur_quality"]
    exp_q = signals["exposure_quality"]

    # Core "issue" score: tuned for demo feel (not medical).
    issue_score = (
        2.6 * redness
        + 1.8 * texture
        + 1.6 * yellowish
        - 0.9 * blur_q
        - 0.7 * exp_q
    )

    # Condition probability.
    sick_prob = _sigmoid(2.4 * issue_score - 0.35)

    # Reduce confidence if image quality is poor.
    quality = 0.55 * blur_q + 0.45 * exp_q
    confidence = float(np.clip(0.55 * sick_prob + 0.45 * quality, 0.0, 1.0))

    if sick_prob < 0.5:
        return HealthPrediction(
            condition="Healthy",
            issue=None,
            confidence=1.0 - confidence,
            signals=signals,
        )

    # Simple issue classification based on dominant signals.
    if redness >= max(yellowish, texture) and redness > 0.06:
        issue = "skin problems"
    elif yellowish >= max(redness, texture) and yellowish > 0.05:
        issue = "infection signs"
    else:
        issue = "weakness indicators"

    return HealthPrediction(
        condition="Sick",
        issue=issue,
        confidence=confidence,
        signals=signals,
    )

