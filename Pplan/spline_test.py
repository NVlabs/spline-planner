import matplotlib.pyplot as plt
import numpy as np
import sympy

import jax
import jax.numpy as jnp
import pdb
def mean_control_effort_coefficients(x0, dx0, xf, dxf):
    """Returns `(c4, c3, c2)` corresponding to `c4 * tf**-4 + c3 * tf**-3 + c2 * tf**-2`."""
    return (12 * (x0 - xf)**2, 12 * (dx0 + dxf) * (x0 - xf), 4 * dx0**2 + 4 * dx0 * dxf + 4 * dxf**2)


def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (x0, dx0, -2 * dx0 / tf - dxf / tf - 3 * x0 / tf**2 + 3 * xf / tf**2,
            dx0 / tf**2 + dxf / tf**2 + 2 * x0 / tf**3 - 2 * xf / tf**3)


def compute_interpolating_spline(state_0, state_f, tf):
    x0, y0, q0, v0 = state_0
    xf, yf, qf, vf = state_f
    dx0, dy0 = v0 * jnp.cos(q0), v0 * jnp.sin(q0)
    dxf, dyf = vf * jnp.cos(qf), vf * jnp.sin(qf)
    return (
        jnp.array(cubic_spline_coefficients(x0, dx0, xf, dxf, tf)),
        jnp.array(cubic_spline_coefficients(y0, dy0, yf, dyf, tf)),
        tf,
    )


def compute_spline_xyvaqt(x_coefficients, y_coefficients, tf, N=30):
    t = jnp.linspace(0, tf, N)
    tp = t[:, None]**np.arange(4)
    dtp = t[:, None]**np.array([0, 0, 1, 2]) * np.arange(4)
    ddtp = t[:, None]**np.array([0, 0, 0, 1]) * np.array([0, 0, 2, 6])
    vx = dtp @ x_coefficients
    vy = dtp @ y_coefficients
    return (
        tp @ x_coefficients,
        tp @ y_coefficients,
        jnp.hypot(dtp @ x_coefficients, dtp @ y_coefficients),
        jnp.hypot(ddtp @ x_coefficients, ddtp @ y_coefficients),
        jnp.arctan2(vy,vx),
        t,
    )
state_0 = np.array([0, 0, 0, 10.])
state_f = np.array([
    np.array([x, y, q, v]) for x in [24, 30, 36] for y in [-4, 0, 4] for q in [-np.pi / 12, 0, np.pi / 12]
    for v in [8, 10, 12]
])
tf = 5.0
xc, yc, tf = jax.vmap(compute_interpolating_spline, in_axes=(None, 0,None))(state_0, state_f, tf)
x, y, v, a, yaw, t = jax.vmap(compute_spline_xyvaqt)(xc, yc, tf)

pdb.set_trace()
plt.figure(figsize=(20, 10))
plt.plot(x.T, y.T)
plt.show()