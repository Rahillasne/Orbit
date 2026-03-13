#!/usr/bin/env python3
"""Benchmark embedding extraction speed across R3M, SigLIP, and OpenCLIP.

Generates synthetic images and measures wall-clock time for each model.
Results are printed and optionally saved to a JSON file.

Usage:
    python scripts/benchmark_embeddings.py [--n-images 100] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image


def _make_images(n: int, size: int = 224) -> list[Image.Image]:
    """Generate N random RGB images."""
    rng = np.random.default_rng(42)
    return [
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
        for _ in range(n)
    ]


def _benchmark_model(model_name: str, images: list[Image.Image], device: str = "cpu") -> dict:
    """Benchmark a single embedding model. Returns timing info."""
    from orbit.embeddings import get_extractor

    try:
        extractor = get_extractor(model_name, device=device)
    except Exception as e:
        return {"model": model_name, "error": str(e), "available": False}

    # Warm up
    try:
        extractor.embed_images(images[:2])
    except Exception as e:
        return {"model": model_name, "error": str(e), "available": False}

    # Benchmark
    start = time.perf_counter()
    embeddings = extractor.embed_images(images)
    elapsed = time.perf_counter() - start

    return {
        "model": model_name,
        "available": True,
        "n_images": len(images),
        "embedding_dim": embeddings.shape[1],
        "total_seconds": round(elapsed, 3),
        "images_per_second": round(len(images) / elapsed, 1),
        "device": device,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--n-images", type=int, default=100, help="Number of images to embed")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    args = parser.parse_args()

    print(f"Generating {args.n_images} synthetic images...")
    images = _make_images(args.n_images)

    models = ["r3m", "siglip", "openclip"]
    results = []

    for model in models:
        print(f"\nBenchmarking {model}...")
        result = _benchmark_model(model, images, device=args.device)
        results.append(result)

        if result.get("available"):
            print(
                f"  {model}: {result['total_seconds']:.3f}s "
                f"({result['images_per_second']:.1f} img/s, "
                f"dim={result['embedding_dim']})"
            )
        else:
            print(f"  {model}: NOT AVAILABLE — {result.get('error', 'unknown')}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    available = [r for r in results if r.get("available")]
    if available:
        fastest = min(available, key=lambda r: r["total_seconds"])
        print(f"Fastest: {fastest['model']} ({fastest['images_per_second']:.1f} img/s)")
    else:
        print("No embedding models available (all in fallback mode)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
