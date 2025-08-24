from typing import Optional, Union

import numpy as np

BCVal = Union[float, np.ndarray]


def _expand_bc(val: Optional[BCVal], f: np.ndarray, axis: int):
    if val is None:
        return None
    val = np.asarray(val)
    # scalar
    if val.shape == ():
        return float(val)
    # make val broadcastable along axis
    # shape = [0] * f.ndim
    # shape[axis] = val.shape[0] if val.ndim > 0 else 0
    return val  # .reshape(shape)


def fdm_1_c(
    f: np.ndarray,
    axis: int = -1,
    bc_left_type: str = "dirichlet",
    bc_left_val: Optional[BCVal] = None,
    bc_right_type: str = "dirichlet",
    bc_right_val: Optional[BCVal] = None,
    periodic: bool = False,
    dx: float = 1.0,
) -> np.ndarray:
    """First derivative (central interior). Left/right BCs independent.
    bc_*_type in {'dirichlet','neumann'}.
    bc_*_val can be scalar or array shaped like f with axis removed.
    If periodic=True, ignores BC arguments and returns periodic central diff.
    """
    f = np.asarray(f)
    nd = f.ndim
    axis = axis % nd
    n = f.shape[axis]
    d = np.empty_like(f, dtype=float)

    if periodic:
        d = 0.5 * (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / dx
        return d

    # interior (central)
    interior_slices = [slice(None)] * nd
    interior_slices[axis] = slice(1, -1)
    d[tuple(interior_slices)] = (
        np.take(f, range(2, n), axis=axis) - np.take(f, range(0, n - 2), axis=axis)
    ) / (2.0 * dx)

    # expand BC values for broadcasting (if array)
    bl = _expand_bc(bc_left_val, f, axis)
    br = _expand_bc(bc_right_val, f, axis)

    # left boundary (index 0)
    left_slice = [slice(None)] * nd
    left_slice[axis] = 0
    if bc_left_type.lower() == "dirichlet":
        g = bl if bl is not None else np.take(f, 0, axis=axis)
        d[tuple(left_slice)] = (np.take(f, 1, axis=axis) - g) / dx
    elif bc_left_type.lower() == "neumann":
        # Neumann supplies derivative value directly
        s = bl if bl is not None else 0.0
        d[tuple(left_slice)] = s
    else:
        raise ValueError("bc_left_type must be 'dirichlet' or 'neumann'")

    # right boundary (index n-1)
    right_slice = [slice(None)] * nd
    right_slice[axis] = -1
    if bc_right_type.lower() == "dirichlet":
        # f'(x_{N-1}) ≈ (g - f_{N-2}) / dx where g is boundary value at rightmost node.
        g = br if br is not None else np.take(f, -1, axis=axis)
        d[tuple(right_slice)] = (g - np.take(f, -2, axis=axis)) / dx
    elif bc_right_type.lower() == "neumann":
        s = br if br is not None else 0.0
        d[tuple(right_slice)] = s
    else:
        raise ValueError("bc_right_type must be 'dirichlet' or 'neumann'")

    return d


def fdm_2_c(
    f: np.ndarray,
    axis: int = -1,
    bc_left_type: str = "dirichlet",
    bc_left_val: Optional[BCVal] = None,
    bc_right_type: str = "dirichlet",
    bc_right_val: Optional[BCVal] = None,
    periodic: bool = False,
    dx: float = 1.0,
) -> np.ndarray:
    """Second derivative (central interior). Left/right BCs independent.
    Uses one-sided high-quality stencils for Dirichlet when possible,
    and ghost-based elimination for Neumann.
    """
    f = np.asarray(f)
    nd = f.ndim
    axis = axis % nd
    n = f.shape[axis]
    d2 = np.empty_like(f, dtype=float)
    h2 = dx * dx

    if periodic:
        d2 = (np.roll(f, -1, axis=axis) - 2.0 * f + np.roll(f, 1, axis=axis)) / h2
        return d2

    # interior (central second difference)
    interior_slices = [slice(None)] * nd
    interior_slices[axis] = slice(1, -1)
    d2[tuple(interior_slices)] = (
        np.take(f, range(2, n), axis=axis)
        - 2.0 * np.take(f, range(1, n - 1), axis=axis)
        + np.take(f, range(0, n - 2), axis=axis)
    ) / h2

    bl = _expand_bc(bc_left_val, f, axis)
    br = _expand_bc(bc_right_val, f, axis)

    # LEFT boundary (index 0)
    left_slice = [slice(None)] * nd
    left_slice[axis] = 0
    if bc_left_type.lower() == "dirichlet":
        g = bl if bl is not None else np.take(f, 0, axis=axis)

        f1 = np.take(f, 1, axis=axis)
        f2 = np.take(f, 2, axis=axis)
        f3 = np.take(f, 3, axis=axis)
        d2[tuple(left_slice)] = (2.0 * g - 5.0 * f1 + 4.0 * f2 - 1.0 * f3) / h2

    elif bc_left_type.lower() == "neumann":
        s = bl if bl is not None else 0.0
        f0 = np.take(f, 0, axis=axis)
        f1 = np.take(f, 1, axis=axis)
        ghost = f1 - 2.0 * s * dx
        d2[tuple(left_slice)] = (f1 - 2.0 * f0 + ghost) / h2
    else:
        raise ValueError("bc_left_type must be 'dirichlet' or 'neumann'")

    # RIGHT boundary (index n-1)
    right_slice = [slice(None)] * nd
    right_slice[axis] = -1
    if bc_right_type.lower() == "dirichlet":
        # Mirror of left: g = f(N-1). One-sided formula:
        g = br if br is not None else np.take(f, -1, axis=axis)
        fm1 = np.take(f, -2, axis=axis)  # f_{N-2}
        fm2 = np.take(f, -3, axis=axis)  # f_{N-3}
        fm3 = np.take(f, -4, axis=axis)  # f_{N-4}
        # f''(x_{N-1}) ≈ (2*g - 5*f_{N-2} + 4*f_{N-3} - f_{N-4}) / dx^2 (one-sided, mirrored)
        d2[tuple(right_slice)] = (2.0 * g - 5.0 * fm1 + 4.0 * fm2 - 1.0 * fm3) / h2

    elif bc_right_type.lower() == "neumann":
        # Right Neumann: s = f'(x_{N-1}); ghost f_{N} = f_{N-2} + 2*s*dx derived from (f_{N} - f_{N-2})/(2 dx) = s
        s = br if br is not None else 0.0
        fNm1 = np.take(f, -1, axis=axis)
        fNm2 = np.take(f, -2, axis=axis)
        ghost = fNm2 + 2.0 * s * dx
        d2[tuple(right_slice)] = (ghost - 2.0 * fNm1 + fNm2) / h2

    else:
        raise ValueError("bc_right_type must be 'dirichlet' or 'neumann'")

    return d2
