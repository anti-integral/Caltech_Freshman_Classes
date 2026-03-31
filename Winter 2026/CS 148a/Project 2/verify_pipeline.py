"""
Verify that your pipeline works end-to-end before submitting.
"""

import sys
import json
import argparse

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download



def verify_pipeline(submission_json: dict, device: str = 'cpu'):
    NUM_CLASSES = 10
    NUM_IMAGES = 10

    # (width, height) pairs — mix of square and non-square, all >= 128
    IMAGE_DIMENSIONS = [
        (128, 128),
        (256, 128),
        (128, 256),
        (200, 200),
        (512, 128),
        (150, 300),
        (300, 150),
        (256, 256),
        (400, 200),
        (200, 400),
    ]

    def generate_test_images(input_channels: int) -> list[tuple[Image.Image, int]]:
        """Generate NUM_IMAGES random PIL images with varying dimensions.

        Returns list of (PIL.Image, label) tuples where label is 0-9.
        """
        rng = np.random.RandomState(42)
        images = []

        for i in range(NUM_IMAGES):
            w, h = IMAGE_DIMENSIONS[i]
            label = i % NUM_CLASSES

            if input_channels == 3:
                arr = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
                img = Image.fromarray(arr, mode='RGB')
            else:
                arr = rng.randint(0, 256, size=(h, w), dtype=np.uint8)
                img = Image.fromarray(arr, mode='L')

            images.append((img, label))

        return images
    
    checks_passed = 0
    checks_failed = 0

    def check(name, passed, detail=""):
        nonlocal checks_passed, checks_failed
        if passed:
            checks_passed += 1
            print(f"  PASS: {name}")
        else:
            checks_failed += 1
            print(f"  FAIL: {name}")
        if detail:
            print(f"        {detail}")

    # --- Step 1: Load submission.json ---
    print("\n[1/6] Loading submission.json...")
    try:
        required_keys = {'username', 'repo_name', 'filename', 'token'}
        missing = required_keys - set(submission_json.keys())
        check("submission.json is valid", not missing,
              f"Missing keys: {missing}" if missing else "")
        if missing:
            print("\nCannot continue without valid submission.json.")
            sys.exit(1)
    except Exception as e:
        check("submission.json is valid", False, str(e))
        sys.exit(1)

    # --- Step 2: Download model from HuggingFace ---
    print("\n[2/6] Downloading model from HuggingFace...")
    try:
        repo_id = f"{submission_json['username']}/{submission_json['repo_name']}"
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=submission_json['filename'],
            token=submission_json['token'],
        )
        check("Model downloaded", True, f"Cached at: {model_path}")
    except Exception as e:
        check("Model downloaded", False, str(e))
        print("\nCannot continue without downloading the model.")
        sys.exit(1)

    # --- Step 3: Load TorchScript model ---
    print("\n[3/6] Loading TorchScript model...")
    try:
        pipeline = torch.jit.load(model_path, map_location=device)
        pipeline.eval()
        check("Model loaded", True)
    except Exception as e:
        check("Model loaded", False, str(e))
        print("\nCannot continue without loading the model.")
        sys.exit(1)

    # --- Step 4: Inspect pipeline attributes ---
    print("\n[4/6] Inspecting pipeline attributes...")

    has_preprocess = hasattr(pipeline, 'preprocess_layers')
    check("Has preprocess_layers", has_preprocess)

    input_channels = 3
    for attr in ['input_channels', 'input_height', 'input_width']:
        try:
            val = getattr(pipeline, attr)
            check(f"Has {attr}", True, f"value = {val}")
            if attr == 'input_channels':
                input_channels = val
        except AttributeError:
            if attr == 'input_channels':
                # Match evaluate_models.py: default to 3 (RGB) when missing
                check(f"Has {attr}", True, "Not found, defaulting to 3 (RGB)")
            else:
                check(f"Has {attr}", False, "Attribute not found on pipeline")

    if not has_preprocess:
        print("\nCannot continue without preprocess_layers.")
        sys.exit(1)

    # --- Step 5: Preprocess images ---
    print(f"\n[5/6] Generating {NUM_IMAGES} test images and preprocessing...")
    convert_to = 'RGB' if input_channels == 3 else 'L'
    test_images = generate_test_images(input_channels)

    tensors = []
    preprocess_ok = True
    for idx, (img, label) in enumerate(test_images):
        w, h = img.size
        try:
            img_converted = img.convert(convert_to)
            tensor = transforms.ToTensor()(img_converted)
            tensor = pipeline.preprocess_layers(tensor)
            tensors.append(tensor)
            print(f"    Image {idx}: {w}x{h} -> tensor {list(tensor.shape)}")
        except Exception as e:
            check(f"Preprocess image {idx} ({w}x{h})", False, str(e))
            preprocess_ok = False

    if not preprocess_ok or not tensors:
        check("All images preprocessed", False)
        print("\nPreprocessing failed. Check your preprocess_layers.")
        sys.exit(1)

    # Check all tensors have the same shape
    shapes = [t.shape for t in tensors]
    all_same = all(s == shapes[0] for s in shapes)
    check("All preprocessed tensors have same shape", all_same,
          f"Shape: {list(shapes[0])}" if all_same else f"Shapes: {[list(s) for s in shapes]}")

    if not all_same:
        print("\npreprocess_layers must resize all images to the same dimensions.")
        sys.exit(1)

    # --- Step 6: Run inference ---
    print("\n[6/6] Running inference...")
    try:
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            preds = pipeline(batch)

        check("Inference ran without error", True)
        check("Output shape is (B,)", preds.shape == (NUM_IMAGES,),
              f"Got shape {list(preds.shape)}, expected [{NUM_IMAGES}]")

        pred_list = preds.cpu().tolist()
        all_valid = all(0 <= p < NUM_CLASSES for p in pred_list)
        check("Predictions in range [0, 9]", all_valid,
              f"Predictions: {pred_list}")

        print(f"\n    Predictions: {pred_list}")

    except Exception as e:
        check("Inference ran without error", False, str(e))

    # --- Summary ---
    total = checks_passed + checks_failed
    print(f"\n{'='*50}")
    print(f"Results: {checks_passed}/{total} checks passed")
    if checks_failed == 0:
        print("Your pipeline is ready for submission!")
    else:
        print(f"{checks_failed} check(s) failed. Fix the issues above before submitting.")
    print(f"{'='*50}\n")

    sys.exit(0 if checks_failed == 0 else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify your pipeline works before submitting.'
    )
    parser.add_argument(
        'submission_json', type=str,
        help='Path to your submission.json file'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Torch device (default: cpu)'
    )
    args = parser.parse_args()

    with open(args.submission_json, 'r') as f:
        submission_json = json.load(f)

    verify_pipeline(submission_json=submission_json, device=args.device)
