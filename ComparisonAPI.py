#!/usr/bin/env python3

from flask import Flask, request, jsonify
from PIL import Image, ImageChops, ImageStat
import io
import traceback
import os
import time

# Optional CLIP imports (lazy / tolerant)
_USE_CLIP = os.getenv("USE_CLIP", "") not in ("", "0", "false", "False")
_CLIP_AVAILABLE = False
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
if _USE_CLIP:
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        _CLIP_AVAILABLE = True
        # model will be loaded lazily on first request to avoid startup overhead
    except Exception:
        _CLIP_AVAILABLE = False

app = Flask(__name__)

# Debug helper: save uploaded bytes to disk for inspection (dev only)
def _save_debug_file(prefix: str, data: bytes) -> str:
    try:
        os.makedirs("received", exist_ok=True)
        fname = os.path.join("received", f"{prefix}_{int(time.time()*1000)}.png")
        with open(fname, "wb") as f:
            f.write(data)
        return fname
    except Exception:
        return "<failed to save>"

def compute_pixel_score(bytes_a: bytes, bytes_b: bytes):
    img_a = Image.open(io.BytesIO(bytes_a)).convert("RGB")
    img_b = Image.open(io.BytesIO(bytes_b)).convert("RGB")

    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, resample=Image.LANCZOS)

    diff = ImageChops.difference(img_a, img_b)
    stat = ImageStat.Stat(diff)
    rms = stat.rms  # per-channel RMS
    mse = sum((r * r) for r in rms) / len(rms)

    max_mse = 255.0 * 255.0
    score = 1.0 - (mse / max_mse)
    if score < 0.0:
        score = 0.0

    return {"score": round(score, 6), "mse": round(mse, 3), "width": img_a.width, "height": img_a.height}

def _ensure_clip_loaded():
    global _CLIP_AVAILABLE, _CLIP_MODEL, _CLIP_PROCESSOR
    if not _CLIP_AVAILABLE:
        return False
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        try:
            _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            # set model to eval mode
            _CLIP_MODEL.eval()
            # if GPU available, you can move to cuda here (optional)
        except Exception:
            _CLIP_AVAILABLE = False
            return False
    return True

def compute_clip_score(bytes_a: bytes, bytes_b: bytes):
    """Return CLIP-based semantic similarity in [0..1], or None if unavailable."""
    if not _USE_CLIP:
        return None
    try:
        # lazy import inside function for safety
        import torch
        from transformers import CLIPProcessor, CLIPModel  # noqa: F401
    except Exception:
        return None

    if not _ensure_clip_loaded():
        return None

    try:
        img_a = Image.open(io.BytesIO(bytes_a)).convert("RGB")
        img_b = Image.open(io.BytesIO(bytes_b)).convert("RGB")

        # prepare inputs (processor will resize/normalize)
        inputs_a = _CLIP_PROCESSOR(images=img_a, return_tensors="pt")
        inputs_b = _CLIP_PROCESSOR(images=img_b, return_tensors="pt")

        with torch.no_grad():
            feats_a = _CLIP_MODEL.get_image_features(**inputs_a)
            feats_b = _CLIP_MODEL.get_image_features(**inputs_b)

            # normalize and cosine similarity
            feats_a = feats_a / feats_a.norm(dim=-1, keepdim=True)
            feats_b = feats_b / feats_b.norm(dim=-1, keepdim=True)
            cos = (feats_a @ feats_b.T).item()  # in [-1,1]
            # map to [0,1]
            clip_score = (cos + 1.0) / 2.0
            return float(clip_score)
    except Exception:
        return None

@app.route("/compare", methods=["POST"])
def compare_endpoint():
    try:
        if "fileA" not in request.files or "fileB" not in request.files:
            return jsonify({"error": "form-data must include 'fileA' and 'fileB'"}), 400

        file_a = request.files["fileA"]
        file_b = request.files["fileB"]

        bytes_a = file_a.read()
        bytes_b = file_b.read()

        # Save debug copies so you can inspect what the comparison service actually received
        saved_a = _save_debug_file("fileA", bytes_a)
        saved_b = _save_debug_file("fileB", bytes_b)

        # Quick checks
        if not bytes_a or not bytes_b:
            return jsonify({
                "error": "uploaded files are empty",
                "fileA_size": len(bytes_a),
                "fileB_size": len(bytes_b),
                "saved_paths": {"fileA": saved_a, "fileB": saved_b}
            }), 400

        # Try opening images early to give clearer errors
        try:
            Image.open(io.BytesIO(bytes_a))
        except Exception as e:
            return jsonify({
                "error": "fileA is not a valid image or cannot be opened by PIL",
                "exception": str(e),
                "saved_path": saved_a
            }), 400

        try:
            Image.open(io.BytesIO(bytes_b))
        except Exception as e:
            return jsonify({
                "error": "fileB is not a valid image or cannot be opened by PIL",
                "exception": str(e),
                "saved_path": saved_b
            }), 400

        # Pixel-based (existing) score
        pixel_result = compute_pixel_score(bytes_a, bytes_b)

        # Optional CLIP-based score (semantic); None if not available
        clip_score = compute_clip_score(bytes_a, bytes_b)
        if clip_score is not None:
            pixel_result["clip_score"] = round(clip_score, 6)

        # include saved paths for dev debugging
        pixel_result["debug"] = {"saved_fileA": saved_a, "saved_fileB": saved_b}
        if _USE_CLIP and not _CLIP_AVAILABLE:
            pixel_result["debug"]["clip"] = "USE_CLIP enabled but CLIP dependencies not available or failed to load"

        return jsonify(pixel_result)
    except Exception:
        tb = traceback.format_exc()
        # print to server console for immediate visibility
        print(tb)
        # return detailed error in response for development debugging
        return jsonify({"error": "internal server error", "traceback": tb}), 500

if __name__ == "__main__":
    # ensure folder exists and warn developer
    os.makedirs("received", exist_ok=True)
    print("ComparisonAPI starting; debug uploads will be saved to ./received")
    if _USE_CLIP and not _CLIP_AVAILABLE:
        print("USE_CLIP=1 but CLIP dependencies not available. Install torch and transformers to enable CLIP.")
    app.run(host="0.0.0.0", port=5000, debug=False)