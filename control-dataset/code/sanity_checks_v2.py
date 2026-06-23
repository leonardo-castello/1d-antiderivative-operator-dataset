import numpy as np

def check_boundary_condition():
    """u(0) must be 0 for every function, at every resolution."""
    print("=== Check 1: boundary condition u(0) = 0 ===")
    for res in [16, 32, 64, 128, 256, 512, 1024]:
        d = np.load(f"../data/test_res{res}.npz")
        max_err = np.max(np.abs(d["u"][:, 0]))
        print(f"  res={res:4d} | max |u(0)| across all functions: {max_err:.2e}")
    print()


def check_resolution_consistency():
    """
    The same 200 test functions should give identical values at grid points
    that are shared across resolutions. x=0 and x=1 are shared by all grids.
    """
    print("=== Check 2: shared grid points match across resolutions ===")
    ref = np.load("../data/test_res1024.npz")

    for res in [16, 32, 64, 128, 256, 512]:
        d = np.load(f"../data/test_res{res}.npz")

        # x=0 and x=1 are always the first and last points
        err_start = np.max(np.abs(d["u"][:, 0]  - ref["u"][:, 0]))
        err_end   = np.max(np.abs(d["u"][:, -1] - ref["u"][:, -1]))
        print(f"  res={res:4d} vs res=1024 | "
              f"max diff at x=0: {err_start:.2e} | "
              f"max diff at x=1: {err_end:.2e}")
    print()


def check_train_test_different():
    """
    Train and test functions were drawn with different seeds.
    Their coefficient arrays should not be equal.
    """
    print("=== Check 3: train and test functions are different ===")
    train = np.load("../data/train_coeffs.npz")
    test  = np.load("../data/test_coeffs.npz")

    # Check shapes
    print(f"  train coeffs shape: {train['coeffs'].shape}")
    print(f"  test  coeffs shape: {test['coeffs'].shape}")

    # Check they are not the same
    overlap = np.allclose(
        train["coeffs"][:200],   # compare first 200 of train vs all of test
        test["coeffs"]
    )
    print(f"  Are first 200 train == test coefficients? {overlap}  (should be False)")
    print()


def check_antiderivative_relationship():
    """
    Numerically differentiate u and check it recovers a.
    Use a high-resolution grid for this so numerical derivative is accurate.
    """
    print("=== Check 4: du/dx ≈ a (antiderivative relationship) ===")
    d = np.load("../data/test_res1024.npz")
    x = d["x"]         # (1024,)
    a = d["a"]          # (200, 1024)
    u = d["u"]          # (200, 1024)

    dx = x[1] - x[0]

    # Central differences for all interior points
    du_dx = (u[:, 2:] - u[:, :-2]) / (2 * dx)   # (200, 1022)
    a_interior = a[:, 1:-1]                        # (200, 1022)

    max_err  = np.max(np.abs(du_dx - a_interior))
    mean_err = np.mean(np.abs(du_dx - a_interior))
    print(f"  max  |du/dx - a|: {max_err:.2e}")
    print(f"  mean |du/dx - a|: {mean_err:.2e}")
    print(f"  (should be very small — float32 + finite difference approximation)")
    print()


if __name__ == "__main__":
    check_boundary_condition()
    check_resolution_consistency()
    check_train_test_different()
    check_antiderivative_relationship()
    print("All checks complete.")