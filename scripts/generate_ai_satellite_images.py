"""
generate_ai_satellite_images.py — AI Satellite Image Generator
==============================================================
Generates AI satellite images using Stable Diffusion and saves them
to data/raw/ with proper metadata entries in metadata.csv.

Requirements (inside your venv):
    pip install diffusers torch accelerate transformers pillow

Usage:
    python3 scripts/generate_ai_satellite_images.py \
        --output_dir  ./data/raw \
        --csv         ./data/metadata.csv \
        --count       1000 \
        --start_id    1001

Notes:
    - On Apple Silicon (M1/M2/M3) this uses the MPS GPU automatically
    - First run downloads the model (~4GB) to ~/.cache/huggingface/
    - Each image takes ~10-30 seconds on Apple Silicon
    - Generated images are labeled 'ai_generated' in the CSV
"""

import argparse
import csv
import os
import random
from datetime import date
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# ---------------------------------------------------------------------------
# Satellite image prompts — varied to create a realistic diverse dataset
# ---------------------------------------------------------------------------

PROMPTS = [
    "satellite view of dense urban city blocks, aerial photography, high resolution",
    "top-down satellite image of suburban neighborhood with streets and houses",
    "aerial satellite view of agricultural farmland with crop fields",
    "satellite image of coastal area with ocean and beach, top-down view",
    "overhead satellite view of mountain terrain with forests",
    "aerial view of industrial area with warehouses and roads",
    "satellite image of river delta with waterways, top-down",
    "overhead view of desert landscape with sand dunes, satellite imagery",
    "satellite view of airport with runways and terminals",
    "aerial satellite image of sports stadium surrounded by parking lots",
    "top-down satellite view of port with ships and docks",
    "satellite image of highway interchange with multiple roads",
    "overhead view of golf course with green fairways, satellite imagery",
    "satellite view of university campus with buildings and paths",
    "aerial image of solar farm with panels, top-down satellite view",
    "satellite view of lake surrounded by forest, overhead photography",
    "top-down aerial view of city park with trees and paths",
    "satellite image of suburban mall with large parking lot",
    "overhead satellite view of residential area with swimming pools",
    "aerial satellite image of train yard with railway tracks",
]

# Negative prompt to improve realism
NEGATIVE_PROMPT = (
    "blurry, low quality, cartoon, illustration, painting, drawing, "
    "text, watermark, logo, distorted, unrealistic colors"
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./data/raw",
                   help="Directory to save generated images")
    p.add_argument("--csv",        default="./data/metadata.csv",
                   help="metadata.csv to append new entries to")
    p.add_argument("--count",      type=int, default=1000,
                   help="Number of images to generate")
    p.add_argument("--start_id",   type=int, default=1001,
                   help="Starting image_id number (set to max existing + 1)")
    p.add_argument("--img_size",   type=int, default=512,
                   help="Output image size in pixels (512 recommended)")
    p.add_argument("--steps",      type=int, default=25,
                   help="Diffusion steps — more = better quality but slower")
    p.add_argument("--model",      default="stabilityai/stable-diffusion-2-1-base",
                   help="Hugging Face model ID to use")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str) -> StableDiffusionPipeline:
    """
    Loads the Stable Diffusion pipeline with the best available device:
      - Apple Silicon (M1/M2/M3): uses MPS (Metal Performance Shaders)
      - NVIDIA GPU:                uses CUDA
      - CPU fallback:              slow but works
    """
    print(f"Loading model: {model_id}")

    # Detect best device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype  = torch.float16
        print("Using Apple Silicon MPS GPU")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        dtype  = torch.float32
        print("WARNING: No GPU found, using CPU — this will be very slow")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,        # disable NSFW filter — not needed for satellite images
        requires_safety_checker=False
    )

    # Use faster DPM scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Memory optimization for Apple Silicon
    if device == "mps":
        pipe.enable_attention_slicing()

    print(f"Model loaded on {device}")
    return pipe

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def get_existing_ids(csv_path: str) -> set:
    """Returns set of existing image_ids in the CSV."""
    if not Path(csv_path).exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["image_id"] for row in reader}

def append_to_csv(csv_path: str, image_id: str, filename: str):
    """Appends a single ai_generated entry to the metadata CSV."""
    file_exists = Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                "image_id", "filename", "source",
                "label", "resolution", "date_collected", "llm_analyzed"
            ])
        writer.writerow([
            image_id,
            filename,
            "stable_diffusion",
            "ai_generated",
            "512x512",
            date.today().isoformat(),
            "false"
        ])

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(pipe: StableDiffusionPipeline, args):
    """Main generation loop."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = get_existing_ids(args.csv)
    generated    = 0
    skipped      = 0
    current_id   = args.start_id

    print(f"\nGenerating {args.count} AI satellite images...")
    print(f"Output dir:  {output_dir}")
    print(f"CSV:         {args.csv}")
    print(f"Start ID:    {args.start_id}")
    print(f"Steps:       {args.steps}")
    print("-" * 50)

    while generated < args.count:
        image_id  = str(current_id).zfill(9)      # → "000001001"
        filename  = f"image{image_id}.jpg"
        out_path  = output_dir / filename

        current_id += 1

        # Skip if already exists
        if image_id in existing_ids or out_path.exists():
            skipped += 1
            continue

        # Pick a random prompt from the list
        prompt = random.choice(PROMPTS)

        try:
            print(f"[{generated + 1}/{args.count}] Generating {filename}")
            print(f"  Prompt: {prompt[:60]}...")

            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=args.img_size,
                height=args.img_size,
                num_inference_steps=args.steps,
                guidance_scale=7.5,
            )

            image: Image.Image = result.images[0]

            # Resize to 224x224 to match training size
            image = image.resize((224, 224), Image.BILINEAR)
            image.save(out_path, "JPEG", quality=95)

            # Append to CSV immediately so progress is saved if interrupted
            append_to_csv(args.csv, image_id, filename)

            generated += 1

            if generated % 50 == 0:
                print(f"\n--- Progress: {generated}/{args.count} generated ---\n")

        except Exception as e:
            print(f"  ERROR generating {filename}: {e}")
            continue

    print(f"\n{'=' * 50}")
    print(f"Done! Generated {generated} images ({skipped} skipped as duplicates)")
    print(f"All entries appended to: {args.csv}")
    print(f"Next run use --start_id {current_id}")
    print(f"{'=' * 50}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Check CSV exists
    if not Path(args.csv).exists():
        print(f"WARNING: CSV not found at {args.csv} — will create a new one")

    # Load pipeline
    pipe = load_pipeline(args.model)

    # Generate
    generate_images(pipe, args)

    # Final summary
    print("\nNext steps:")
    print("  1. Re-run train_model.py with the updated metadata.csv")
    print("  2. Check class balance — you want roughly 50/50 real vs ai_generated")
    print("  3. Run Main.java to test the retrained model")


if __name__ == "__main__":
    main()
