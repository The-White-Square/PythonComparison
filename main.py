from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
from PIL import Image
import io

import torch
import open_clip

import cv2
import numpy as np

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Score mapping (tune these with real game samples)
COSINE_LOW = 0.25
COSINE_HIGH = 0.65

# Edge coverage thresholds (lowered to be less strict)
EDGE_COV_HARD_ZERO = 0.0015   # below this: score = 0
EDGE_COV_SOFT_CAP = 0.0025    # below this: cap score (less strict)
SOFT_CAP_MAX_SCORE = 40.0     # raised from 15.0

OUT_SIZE = 512  # for sketch images

# weights for combined cosine (semantic-ish)
W_SKETCH = 0.6
W_RAW = 0.4

# weights for final percent score: CLIP-based vs geometry-based
W_SCORE_CLIP = 0.55
W_SCORE_EDGE = 0.45

# Edge-F1 tolerance in pixels (bigger = more forgiving)
EDGE_F1_TOL_PX = 2

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = FastAPI(title="Image Comparison API")

# ------------------------------------------------------------
# Load model once
# ------------------------------------------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
model = model.to(DEVICE).eval()

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _open_image_or_400(data: bytes, which: str) -> Image.Image:
    if not data:
        raise HTTPException(status_code=400, detail=f"{which} is empty")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{which} is not a valid image: {e}")


def letterbox_to_square_rgb(pil_img: Image.Image, size: int, bg=(255, 255, 255)) -> Image.Image:
    img = pil_img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    square = Image.new("RGB", (s, s), bg)
    square.paste(img, ((s - w) // 2, (s - h) // 2))
    return square.resize((size, size))


def to_sketch_pil(pil_img: Image.Image) -> Image.Image:
    """Convert image to a sketch-like black-on-white representation, ignoring pure white."""
    square = letterbox_to_square_rgb(pil_img, OUT_SIZE)
    arr = np.array(square)
    
    # Create mask for non-white pixels (anything that's not pure 255,255,255)
    non_white_mask = ~((arr[:,:,0] == 255) & (arr[:,:,1] == 255) & (arr[:,:,2] == 255))
    
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(gray, threshold1=60, threshold2=160)
    
    # Only keep edges where there's non-white content
    edges = edges & non_white_mask.astype(np.uint8) * 255
    
    inv = 255 - edges  # black lines on white background

    # Thicken slightly so it resembles marker strokes
    kernel = np.ones((2, 2), np.uint8)
    inv = cv2.erode(inv, kernel, iterations=1)

    return Image.fromarray(inv).convert("RGB")


def edge_coverage(pil_img: Image.Image) -> float:
    """Fraction of pixels that are edges (rough 'how much did they draw?'), ignoring white."""
    arr = np.array(pil_img.convert("RGB"))
    
    # Mask out pure white pixels
    non_white_mask = ~((arr[:,:,0] == 255) & (arr[:,:,1] == 255) & (arr[:,:,2] == 255))
    
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    
    # Only count edges where there's non-white content
    edges = edges & non_white_mask.astype(np.uint8) * 255
    
    return float((edges > 0).mean())


def cosine_to_percent(cos: float, low: float = COSINE_LOW, high: float = COSINE_HIGH) -> float:
    """Map cosine similarity to 0..100 with clamping. Tune low/high."""
    t = (cos - low) / (high - low)
    t = max(0.0, min(1.0, t))
    return t * 100.0


def clip_cosine_for_two_images(img1: Image.Image, img2: Image.Image) -> float:
    """Compute normalized CLIP image embedding cosine similarity."""
    t = torch.stack([preprocess(img1), preprocess(img2)]).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return float((feats[0] @ feats[1]).item())


def _binary_ink_mask_from_sketch(sketch_rgb: Image.Image) -> np.ndarray:
    """
    Convert sketch RGB (black lines on white) into binary ink mask.
    1 = ink (non-white content), 0 = background.
    """
    arr = np.array(sketch_rgb)
    # Consider anything that's not pure white as ink
    ink = ~((arr[:,:,0] == 255) & (arr[:,:,1] == 255) & (arr[:,:,2] == 255))
    return ink.astype(np.uint8)


def edge_f1(orig_sketch: Image.Image, draw_sketch: Image.Image, tol_px: int = EDGE_F1_TOL_PX) -> float:
    """
    F1 score between ink masks, with tolerance using dilation.
    Returns 0..1. Higher means strokes overlap spatially.
    """
    a = _binary_ink_mask_from_sketch(orig_sketch)
    b = _binary_ink_mask_from_sketch(draw_sketch)

    if a.sum() == 0 or b.sum() == 0:
        return 0.0

    k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
    a_d = cv2.dilate(a, k, iterations=1)
    b_d = cv2.dilate(b, k, iterations=1)

    # precision: of drawn ink, how much is near original ink?
    tp_p = (b & a_d).sum()
    prec = tp_p / (b.sum() + 1e-9)

    # recall: of original ink, how much is near drawn ink?
    tp_r = (a & b_d).sum()
    rec = tp_r / (a.sum() + 1e-9)

    return float((2 * prec * rec) / (prec + rec + 1e-9))


# ------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------
@app.post("/compare")
async def compare_images(
    # old (Flask) field names
    fileA: Optional[UploadFile] = File(default=None),
    fileB: Optional[UploadFile] = File(default=None),
    # optional newer names
    original: Optional[UploadFile] = File(default=None),
    drawing: Optional[UploadFile] = File(default=None),
):
    # Prefer old names to preserve existing integration
    a = fileA or original
    b = fileB or drawing

    if a is None or b is None:
        raise HTTPException(
            status_code=400,
            detail="form-data must include either ('fileA' and 'fileB') or ('original' and 'drawing')",
        )

    bytes_a = await a.read()
    bytes_b = await b.read()

    orig_img = _open_image_or_400(bytes_a, "original/fileA")
    draw_img = _open_image_or_400(bytes_b, "drawing/fileB")

    # --- Sketches (used for geometry + also for CLIP-sketch similarity) ---
    orig_sketch = to_sketch_pil(orig_img)
    draw_sketch = to_sketch_pil(draw_img)

    # --- Complexity check (prevents 'one line' high scores) ---
    cov = edge_coverage(draw_sketch)

    # --- Raw images: letterbox to reduce CLIP center-crop issues ---
    orig_raw = letterbox_to_square_rgb(orig_img, 512)
    draw_raw = letterbox_to_square_rgb(draw_img, 512)

    # --- Two CLIP cosines (semantic-ish) ---
    cos_sketch = clip_cosine_for_two_images(orig_sketch, draw_sketch)
    cos_raw = clip_cosine_for_two_images(orig_raw, draw_raw)
    cos_final = W_SKETCH * cos_sketch + W_RAW * cos_raw

    # --- Geometry score: edge F1 (structure match) ---
    f1 = edge_f1(orig_sketch, draw_sketch, tol_px=EDGE_F1_TOL_PX)  # 0..1
    edge_score = f1

    # --- Combine to final percent ---
    clip_percent = cosine_to_percent(cos_final)
    score = W_SCORE_CLIP * clip_percent + W_SCORE_EDGE * edge_score

    # Apply penalties/caps for very low drawing complexity
    if cov < EDGE_COV_HARD_ZERO:
        score = 0.0
    elif cov < EDGE_COV_SOFT_CAP:
        score = min(score, SOFT_CAP_MAX_SCORE)

    # Return simplified response that matches controller expectations
    return {
        "score": round(score, 2),  # Controller looks for "score"
        "message": f"Similarity: {score:.1f}%",
        "details": {
            "clip_similarity": round(clip_percent, 2),
            "edge_match": round(edge_score, 2),
            "coverage": round(cov, 6)
        }
    }
