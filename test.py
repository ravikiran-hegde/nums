# %%

import numpy as np

from fdm import fdm_1_c, fdm_2_c


def test_1d():
    import numpy as np

    print("=== 1D tests ===")
    N = 1000
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]

    # Function: sin(x)
    f = np.sin(x)
    df_exact = np.cos(x)
    d2f_exact = -np.sin(x)

    # Periodic BCs
    df_num = fdm_1_c(f, axis=0, bcs="periodic", dx=dx)
    d2f_num = fdm_2_c(f, axis=0, bcs="periodic", dx=dx)

    print("1D periodic, first derivative max error:", np.max(np.abs(df_num - df_exact)))
    print(
        "1D periodic, second derivative max error:", np.max(np.abs(d2f_num - d2f_exact))
    )

    # Dirichlet BCs: f(0)=0, f(L)=0 for sin(x)
    df_num_dir = fdm_1_c(f, axis=0, bcs="dirichlet", bc_values=(0.0, 0.0), dx=dx)
    d2f_num_dir = fdm_2_c(f, axis=0, bcs="dirichlet", bc_values=(0.0, 0.0), dx=dx)
    print(
        "1D dirichlet, first derivative max error:",
        np.max(np.abs(df_num_dir[1:-1] - df_exact[1:-1])),
    )
    print(
        "1D dirichlet, second derivative max error:",
        np.max(np.abs(d2f_num_dir[1:-1] - d2f_exact[1:-1])),
    )

    # Neumann BCs: df/dx(0)=1, df/dx(L)=-1 for sin(x)
    df_exact_neu_left = np.cos(0)  # = 1
    df_exact_neu_right = np.cos(L)  # = 1 actually for sin(x), not -1
    df_num_neu = fdm_1_c(
        f,
        axis=0,
        bcs="neumann",
        bc_values=(df_exact_neu_left, df_exact_neu_right),
        dx=dx,
    )
    d2f_num_neu = fdm_2_c(
        f,
        axis=0,
        bcs="neumann",
        bc_values=(df_exact_neu_left, df_exact_neu_right),
        dx=dx,
    )
    print(
        "1D neumann, first derivative max error:",
        np.max(np.abs(df_num_neu[1:-1] - df_exact[1:-1])),
    )
    print(
        "1D neumann, second derivative max error:",
        np.max(np.abs(d2f_num_neu[1:-1] - d2f_exact[1:-1])),
    )


def test_2d():
    import numpy as np

    print("\n=== 2D tests ===")
    Nx, Ny = 50, 60
    Lx, Ly = 2 * np.pi, 2 * np.pi
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    # Function: sin(x) * cos(y)
    f = np.sin(X) * np.cos(Y)
    dfdx_exact = np.cos(X) * np.cos(Y)
    dfdy_exact = -np.sin(X) * np.sin(Y)
    d2fdx_exact = -np.sin(X) * np.cos(Y)
    d2fdy_exact = -np.sin(X) * np.cos(Y)

    # Periodic in both directions
    dfdx_num = fdm_1_c(f, axis=0, bcs="periodic", dx=dx)
    dfdy_num = fdm_1_c(f, axis=1, bcs="periodic", dx=dy)
    d2fdx_num = fdm_2_c(f, axis=0, bcs="periodic", dx=dx)
    d2fdy_num = fdm_2_c(f, axis=1, bcs="periodic", dx=dy)

    print("2D periodic, df/dx max error:", np.max(np.abs(dfdx_num - dfdx_exact)))
    print("2D periodic, df/dy max error:", np.max(np.abs(dfdy_num - dfdy_exact)))
    print("2D periodic, d2f/dx2 max error:", np.max(np.abs(d2fdx_num - d2fdx_exact)))
    print("2D periodic, d2f/dy2 max error:", np.max(np.abs(d2fdy_num - d2fdy_exact)))


if __name__ == "__main__":
    import numpy as np

    from fdm import fdm_1_c, fdm_2_c

    test_1d()
    test_2d()

# %%
# ---- Tests ----

import numpy as np

from fdm1 import fdm_1_c, fdm_2_c


def approx_allclose(a, b, rtol=1e-6, atol=1e-8):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        diff = np.abs(a - b)
        i = np.unravel_index(np.argmax(diff), diff.shape)
        raise AssertionError(
            f"Max diff {diff.max():.3e} at index {i}. rtol={rtol} atol={atol}\n"
            f"Computed: {a[i]!r}\nExpected: {b[i]!r}"
        )


def test_periodic_sin():
    # periodic test for first and second derivative
    N = 1000
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = 3.0
    f = np.sin(k * x)
    dx = x[1] - x[0]

    d1_expected = k * np.cos(k * x)
    d2_expected = -k * k * np.sin(k * x)

    d1 = fdm_1_c(f, axis=-1, periodic=True, dx=dx)
    d2 = fdm_2_c(f, axis=-1, periodic=True, dx=dx)

    approx_allclose(d1, d1_expected, rtol=1e-6, atol=1e-6)
    approx_allclose(d2, d2_expected, rtol=1e-5, atol=1e-5)


def test_dirichlet_quadratic():
    # Dirichlet BCs using f(x) = x^2 -> f' = 2x, f'' = 2
    N = 1000
    x = np.linspace(0.0, 1.0, N)
    dx = x[1] - x[0]
    f = x**2

    # exact BC values
    left_val = f[0]
    right_val = f[-1]

    d1_expected = 2.0 * x
    d2_expected = 2.0 * np.ones_like(x)

    d1 = fdm_1_c(
        f,
        axis=-1,
        bc_left_type="dirichlet",
        bc_left_val=left_val,
        bc_right_type="dirichlet",
        bc_right_val=right_val,
        dx=dx,
    )
    d2 = fdm_2_c(
        f,
        axis=-1,
        bc_left_type="dirichlet",
        bc_left_val=left_val,
        bc_right_type="dirichlet",
        bc_right_val=right_val,
        dx=dx,
    )

    approx_allclose(
        d1, d1_expected, rtol=1e-4, atol=1e-6
    )  # first-ord error at boundaries
    approx_allclose(
        d2, d2_expected, rtol=1e-3, atol=1e-6
    )  # boundaries lower acc, interior exact-ish


def test_neumann_quadratic():
    # Neumann BCs on f(x)=x^2 with slope 2x at boundaries
    N = 1000
    x = np.linspace(0.0, 1.0, N)
    dx = x[1] - x[0]
    f = x**2

    left_slope = 2.0 * x[0]
    right_slope = 2.0 * x[-1]

    d1_expected = 2.0 * x
    d2_expected = 2.0 * np.ones_like(x)

    d1 = fdm_1_c(
        f,
        axis=-1,
        bc_left_type="neumann",
        bc_left_val=left_slope,
        bc_right_type="neumann",
        bc_right_val=right_slope,
        dx=dx,
    )
    d2 = fdm_2_c(
        f,
        axis=-1,
        bc_left_type="neumann",
        bc_left_val=left_slope,
        bc_right_type="neumann",
        bc_right_val=right_slope,
        dx=dx,
    )

    approx_allclose(d1, d1_expected, rtol=1e-4, atol=1e-6)
    approx_allclose(d2, d2_expected, rtol=1e-3, atol=1e-6)


def test_mixed_bc():
    # left Dirichlet, right Neumann on cubic-ish function
    N = 1000
    x = np.linspace(0.0, 2.0, N)
    dx = x[1] - x[0]
    f = np.exp(0.5 * x)  # arbitrary smooth function

    left_val = f[0]
    right_slope = 0.5 * np.exp(
        0.5 * x[-1]
    )  # derivative of exp(0.5 x) is 0.5 exp(0.5 x)

    d1_expected = 0.5 * np.exp(0.5 * x)
    d2_expected = 0.25 * np.exp(0.5 * x)

    d1 = fdm_1_c(
        f,
        axis=-1,
        bc_left_type="dirichlet",
        bc_left_val=left_val,
        bc_right_type="neumann",
        bc_right_val=right_slope,
        dx=dx,
    )
    d2 = fdm_2_c(
        f,
        axis=-1,
        bc_left_type="dirichlet",
        bc_left_val=left_val,
        bc_right_type="neumann",
        bc_right_val=right_slope,
        dx=dx,
    )

    approx_allclose(d1, d1_expected, rtol=1e-4, atol=1e-6)
    approx_allclose(d2, d2_expected, rtol=1e-3, atol=1e-6)


def test_array_bcs_multicell():
    # Broadcast/array BC support: create multiple transverse cells
    N = 1000
    x = np.linspace(0.0, 2.0 * np.pi, N)
    dx = x[1] - x[0]
    ks = np.array([1.0, 2.0, 3.0])
    A = np.array([1.0, 0.5, 2.0])

    # create shape (len(A), N)
    f = np.vstack([A[i] * np.sin(ks[i] * x) for i in range(len(A))])
    # exact derivatives
    d1_expected = np.vstack([A[i] * ks[i] * np.cos(ks[i] * x) for i in range(len(A))])
    d2_expected = np.vstack(
        [-A[i] * (ks[i] ** 2) * np.sin(ks[i] * x) for i in range(len(A))]
    )

    # Provide vectorized BCs: left and right arrays of shape (len(A),)
    left_vals = f[:, 0].copy()
    right_vals = f[:, -1].copy()
    left_slopes = d1_expected[:, 0].copy()
    right_slopes = d1_expected[:, -1].copy()

    # Dirichlet array BCs
    d1_dir = fdm_1_c(
        f,
        axis=-1,
        bc_left_type="dirichlet",
        bc_left_val=left_vals,
        bc_right_type="dirichlet",
        bc_right_val=right_vals,
        dx=dx,
    )
    d2_dir = fdm_2_c(
        f,
        axis=-1,
        bc_left_type="dirichlet",
        bc_left_val=left_vals,
        bc_right_type="dirichlet",
        bc_right_val=right_vals,
        dx=dx,
    )

    approx_allclose(d1_dir, d1_expected, rtol=2e-6, atol=1e-5)
    approx_allclose(d2_dir, d2_expected, rtol=2e-5, atol=1e-5)

    # Neumann array BCs
    d1_neu = fdm_1_c(
        f,
        axis=-1,
        bc_left_type="neumann",
        bc_left_val=left_slopes,
        bc_right_type="neumann",
        bc_right_val=right_slopes,
        dx=dx,
    )
    d2_neu = fdm_2_c(
        f,
        axis=-1,
        bc_left_type="neumann",
        bc_left_val=left_slopes,
        bc_right_type="neumann",
        bc_right_val=right_slopes,
        dx=dx,
    )

    approx_allclose(d1_neu, d1_expected, rtol=2e-6, atol=1e-5)
    approx_allclose(d2_neu, d2_expected, rtol=2e-4, atol=1e-5)


if __name__ == "__main__":
    tests = [
        test_periodic_sin,
        test_dirichlet_quadratic,
        test_neumann_quadratic,
        test_mixed_bc,
        test_array_bcs_multicell,
    ]
    failures = []
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"{name}: PASS")
        except AssertionError as e:
            print(f"{name}: FAIL\n{e}")
            failures.append((name, e))
        except Exception as e:
            print(f"{name}: ERROR\n{e}")
            failures.append((name, e))

    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        raise SystemExit(1)
    else:
        print("\nAll tests passed.")

# %%
from fdm import fdm_2_c as fdm_py
from fdm1 import fdm_2_c as fdm1_py


def test_accuracy():
    dx = 0.1
    x = np.arange(0, 2 * np.pi, dx)  # grid for periodic
    f = np.sin(x)
    f_true = -np.sin(x)

    # Periodic BC
    d1 = fdm_py(f, axis=0, bcs="periodic", dx=dx)
    d2 = fdm1_py(f, axis=0, periodic=True, dx=dx)
    err1 = np.linalg.norm(d1 - f_true)
    err2 = np.linalg.norm(d2 - f_true)
    print("Periodic:", err1, err2)

    # Dirichlet BC (f(0)=0, f(L)=0) with f=x^2
    x = np.linspace(0, 1, 11)
    f = x**2
    f_true = 2 * np.ones_like(x)

    d1 = fdm_py(f, axis=0, bcs="dirichlet", bc_values=(0, 1), dx=x[1] - x[0])
    d2 = fdm1_py(
        f,
        axis=0,
        bc_left_type="dirichlet",
        bc_left_val=0,
        bc_right_type="dirichlet",
        bc_right_val=1,
        dx=x[1] - x[0],
    )
    print("Dirichlet:", np.linalg.norm(d1 - f_true), np.linalg.norm(d2 - f_true))

    # Neumann BC (f'(0)=0, f'(1)=2x|x=1=2) with f=x^2
    f_true = 2 * np.ones_like(x)
    d1 = fdm_py(f, axis=0, bcs="neumann", bc_values=(0, 2), dx=x[1] - x[0])
    d2 = fdm1_py(
        f,
        axis=0,
        bc_left_type="neumann",
        bc_left_val=0,
        bc_right_type="neumann",
        bc_right_val=2,
        dx=x[1] - x[0],
    )
    print("Neumann:", np.linalg.norm(d1 - f_true), np.linalg.norm(d2 - f_true))


if __name__ == "__main__":
    test_accuracy()

# %%
from fdm import fdm_2_c as fdm_old
from fdm1 import fdm_2_c as fdm_new


# Analytical functions
def f_periodic(x):
    return np.sin(x)


def f_periodic_ddx(x):
    return -np.sin(x)


def f_cubic(x):
    return x**3


def f_cubic_ddx(x):
    return 6 * x


# L2 error helper
def l2_error(u_num, u_exact):
    return np.sqrt(np.mean((u_num - u_exact) ** 2))


# Test settings
dx_values = [0.1, 0.05, 0.025, 0.0125]

# Store results
errors = {"periodic": [], "dirichlet": [], "neumann": []}

for dx in dx_values:
    # --- Periodic ---
    x = np.arange(0, 2 * np.pi, dx)  # no duplicate endpoint
    f = f_periodic(x)
    f_ddx_true = f_periodic_ddx(x)

    d2_old = fdm_old(f, dx=dx, bcs="periodic")
    d2_new = fdm_new(f, dx=dx, periodic=True)

    errors["periodic"].append(
        (l2_error(d2_old, f_ddx_true), l2_error(d2_new, f_ddx_true))
    )

    # --- Dirichlet ---
    x = np.arange(0, 1 + dx, dx)
    f = f_cubic(x)
    f_ddx_true = f_cubic_ddx(x)

    d2_old = fdm_old(f, dx=dx, bcs="dirichlet", bc_values=(0, 1))
    d2_new = fdm_new(
        f,
        dx=dx,
        bc_left_type="dirichlet",
        bc_left_val=0,
        bc_right_type="dirichlet",
        bc_right_val=1,
    )

    errors["dirichlet"].append(
        (l2_error(d2_old, f_ddx_true), l2_error(d2_new, f_ddx_true))
    )

    # --- Neumann ---
    x = np.arange(0, 1 + dx, dx)
    f = f_cubic(x)
    f_ddx_true = f_cubic_ddx(x)

    # f'(0)=0, f'(1)=3
    d2_old = fdm_old(f, dx=dx, bcs="neumann", bc_values=(0, 3))
    d2_new = fdm_new(
        f,
        dx=dx,
        bc_left_type="neumann",
        bc_left_val=0,
        bc_right_type="neumann",
        bc_right_val=3,
    )

    errors["neumann"].append(
        (l2_error(d2_old, f_ddx_true), l2_error(d2_new, f_ddx_true))
    )

# Print results
for bc_type in errors:
    print(f"\nBC: {bc_type}")
    print("dx      fdm_old        fdm_new")
    for i, dx in enumerate(dx_values):
        e_old, e_new = errors[bc_type][i]
        print(f"{dx:<6} {e_old:<14.6e} {e_new:<14.6e}")

# %%
import numpy as np
from fdm import fdm_2_c as fdm_old
from fdm1 import fdm_2_c as fdm_new

# Grid
nx, nz = 50, 50
Lx, Lz = 2*np.pi, 1.0
dx, dz = Lx/nx, Lz/nz
x = np.linspace(0, Lx-dx, nx)  # periodic in x
z = np.linspace(0, Lz, nz)     # Dirichlet/Neumann in z
X, Z = np.meshgrid(x, z, indexing='ij')

# Analytical function
F = np.sin(X) * (Z**2 + Z**3)
F_xx_true = -np.sin(X) * (Z**2 + Z**3)
F_zz_true = np.sin(X) * (2 + 6*Z)

# --- x direction (periodic) ---
d2_old = np.empty_like(F)
d2_new = np.empty_like(F)
for i in range(nz):
    f = F[:, i]
    d2_old[:, i] = fdm_old(f, dx=dx, bcs="periodic")
    d2_new[:, i] = fdm_new(f, dx=dx, periodic=True)

print("2D periodic x L2 errors:")
print("fdm_old:", np.sqrt(np.mean((d2_old - F_xx_true)**2)))
print("fdm_new:", np.sqrt(np.mean((d2_new - F_xx_true)**2)))

# --- z direction: Dirichlet BCs (consistent with F) ---
d2_old = np.empty_like(F)
d2_new = np.empty_like(F)
for i in range(nx):
    f = F[i, :]
    d2_old[i, :] = fdm_old(f, dx=dz, bcs="dirichlet", bc_values=(f[0], f[-1]))
    d2_new[i, :] = fdm_new(f, dx=dz,
                            bc_left_type="dirichlet", bc_left_val=f[0],
                            bc_right_type="dirichlet", bc_right_val=f[-1])

print("2D Dirichlet z L2 errors:")
print("fdm_old:", np.sqrt(np.mean((d2_old - F_zz_true)**2)))
print("fdm_new:", np.sqrt(np.mean((d2_new - F_zz_true)**2)))

# --- z direction: Neumann BCs (consistent with F) ---
d2_old = np.empty_like(F)
d2_new = np.empty_like(F)
F_z_true = np.sin(X) * (2*Z + 3*Z**2)  # derivative in z

for i in range(nx):
    f = F[i, :]
    df0 = F_z_true[i, 0]
    dfL = F_z_true[i, -1]
    d2_old[i, :] = fdm_old(f, dx=dz, bcs="neumann", bc_values=(df0, dfL))
    d2_new[i, :] = fdm_new(f, dx=dz,
                            bc_left_type="neumann", bc_left_val=df0,
                            bc_right_type="neumann", bc_right_val=dfL)

print("2D Neumann z L2 errors:")
print("fdm_old:", np.sqrt(np.mean((d2_old - F_zz_true)**2)))
print("fdm_new:", np.sqrt(np.mean((d2_new - F_zz_true)**2)))


# %%
