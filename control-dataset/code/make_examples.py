import numpy as np
import matplotlib.pyplot as plt
import os
from math_core import sample_coefficients, evaluate, make_grid, K_MODES, DECAY_POWER

def main():
    os.makedirs("../examples", exist_ok=True)
    plt.rcParams.update({"font.size": 11})

    # Figure 1: examples (a, u) pairs at train resolution (64)
    rng = np.random.default_rng(1)
    coeffs, phases = sample_coefficients(4, K_MODES, DECAY_POWER, rng)
    x = make_grid(64)
    a, u = evaluate(coeffs, phases, x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(4):
        axes[0].plot(x, a[i], label=f"sample {i+1}")
        axes[1].plot(x, u[i], label=f"sample {i+1}")
    axes[0].set_title("Input functions  a(x)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("a(x)")
    axes[1].set_title("Output functions  u(x) = ∫₀ˣ a(s) ds")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle("Example input/output pairs at training resolution (s=64)")
    fig.tight_layout()
    fig.savefig("../examples/fig1_example_pairs.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: same function at multiple resolutions (resolution-invariance) ---
    rng2 = np.random.default_rng(99)
    coeffs2, phases2 = sample_coefficients(1, K_MODES, DECAY_POWER, rng2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for res, style in zip([16, 64, 1024], ["o-", "s-", "-"]):
        xg = make_grid(res)
        a2, u2 = evaluate(coeffs2, phases2, xg)
        # Only show markers for the lower resolutions so the 1024 line is clean
        ms = 4 if res <= 64 else 0
        axes[0].plot(xg, a2[0], style, markersize=ms, label=f"s = {res}")
        axes[1].plot(xg, u2[0], style, markersize=ms, label=f"s = {res}")
    axes[0].set_title("a(x): same function, 3 resolutions")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("a(x)")
    axes[1].set_title("u(x): same function, 3 resolutions")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[0].legend(); axes[1].legend()
    fig.suptitle("Same (a, u) pair sampled at s=16, 64, 1024")
    fig.tight_layout()
    fig.savefig("../examples/fig2_resolution_invariance.png", dpi=150)
    plt.close(fig)

    print("Saved fig1_example_pairs.png and fig2_resolution_invariance.png")

if __name__ == "__main__":    
    main()