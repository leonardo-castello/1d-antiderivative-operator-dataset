import numpy as np
import json 
import os
from math_core import sample_coefficients, evaluate, make_grid, K_MODES, DECAY_POWER

def generate_split(n_samples, resolution, seed):
    """
    Sample n_samples functions (using given seed), then evaluate them
    exactly at every resolution in the list.
    
    Returns a dict with the raw coefficients/phases, plus one entry
    per resolution containing x, a, u arrays.
    """

    rng = np.random.default_rng(seed)
    coeffs, phases = sample_coefficients(n_samples, K_MODES, DECAY_POWER, rng)

    split = {
        "coeffs": coeffs,
        "phases": phases,
        "seed": seed,
    }

    for res in resolution:
        x = make_grid(res)
        a, u = evaluate(coeffs, phases, x)
        split[res] = {
            "x": x,
            "a": a,
            "u": u,
        }

    return split

def save_split(split, resolution, out_dir, prefix):
    """
    Save each resolution to its own .npz file, plus the raw
    coefficients separately for full reproducibility.
    """

    os.makedirs(out_dir, exist_ok=True)
    
    # Save raw coefficients and phases so any resolution can be regenerated
    np.savez(os.path.join(out_dir, f"{prefix}_coeffs.npz"), coeffs=split["coeffs"], phases=split["phases"], seed=split["seed"])

    # Save each resolution separately
    for res in resolution:
        np.savez(os.path.join(out_dir, f"{prefix}_res{res}.npz"), x=split[res]["x"], a=split[res]["a"], u=split[res]["u"])

def main():
    TRAIN_SEED = 0
    TEST_SEED     = 12345
    N_TRAIN       = 1000
    N_TEST        = 200
    TRAIN_RES     = 64
    TEST_RES_LIST = [16, 32, 64, 128, 256, 512, 1024]
    OUT_DIR       = "../data"

    print("Generating train split...")
    train = generate_split(N_TRAIN, [TRAIN_RES], TRAIN_SEED)
    save_split(train, [TRAIN_RES], OUT_DIR, "train")
    print(f"  Saved {N_TRAIN} functions at resolution {TRAIN_RES}")

    print("Generating test split...")
    test = generate_split(N_TEST, TEST_RES_LIST, TEST_SEED)
    save_split(test, TEST_RES_LIST, OUT_DIR, "test")
    print(f"  Saved {N_TEST} functions at resolutions {TEST_RES_LIST}")

    # Save metadata for reproducibility
    meta = {
        "operator":        "antiderivative: u(x) = integral_0^x a(s) ds",
        "domain":          [0.0, 1.0],
        "K_modes":         K_MODES,
        "decay_power":     DECAY_POWER,
        "n_train":         N_TRAIN,
        "train_seed":      TRAIN_SEED,
        "train_res":       TRAIN_RES,
        "n_test":          N_TEST,
        "test_seed":       TEST_SEED,
        "test_res_list":   TEST_RES_LIST,
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved metadata.json")
    print("Done...")

if __name__ == "__main__":
    main()