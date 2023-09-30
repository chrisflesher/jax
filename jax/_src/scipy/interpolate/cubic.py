# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import typing

import jax
import jax.numpy as jnp
import scipy.spatial.interpolate
from jax._src.numpy.util import _wraps

from . import PPoly


@_wraps(scipy.interpolate.CubicHermiteSpline)
class CubicHermiteSpline(PPoly):
  """Piecewise-cubic interpolator matching values and first derivatives."""

  axis: int

  def __init__(cls, x: jax.Array, y: jax.Array, dydx: jax.Array, axis: int = 0, extrapolate: typing.Optional[bool] = None):
    if extrapolate is None:
      extrapolate = True
    x, dx, y, axis, dydx = _prepare_input(x, y, axis, dydx)
    dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
    slope = jnp.diff(y, axis=0) / dxr
    t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr
    c = jnp.vstack((t / dxr, (slope - dydx[:-1]) / dxr - t, dydx[:-1], y[:-1]))
    super.__init__(c, x, extrapolate=extrapolate)
    self.axis = axis


def _prepare_input(x: jax.Array, y: jax.Array, axis: int, dydx: typing.Optional[jax.Array] = None):
  if jnp.issubdtype(x.dtype, jnp.complexfloating):
    raise ValueError("`x` must contain real values.")
  x = x.astype(float)
  if jnp.issubdtype(y.dtype, jnp.complexfloating):
    dtype = complex
  else:
    dtype = float
  if dydx is not None:
    dydx = jnp.asarray(dydx)
    if y.shape != dydx.shape:
      raise ValueError("The shapes of `y` and `dydx` must be identical.")
    if jnp.issubdtype(dydx.dtype, jnp.complexfloating):
      dtype = complex
    dydx = dydx.astype(dtype, copy=False)
  y = y.astype(dtype, copy=False)
  axis = axis % y.ndim
  if x.ndim != 1:
    raise ValueError("`x` must be 1-dimensional.")
  if x.shape[0] < 2:
    raise ValueError("`x` must contain at least 2 elements.")
  if x.shape[0] != y.shape[axis]:
    raise ValueError(f"The length of `y` along `axis`={axis} doesn't match the length of `x`")
  if not jnp.all(jnp.isfinite(x)):
    raise ValueError("`x` must contain only finite values.")
  if not jnp.all(jnp.isfinite(y)):
    raise ValueError("`y` must contain only finite values.")
  if dydx is not None and not jnp.all(jnp.isfinite(dydx)):
    raise ValueError("`dydx` must contain only finite values.")
  dx = jnp.diff(x)
  if jnp.any(dx <= 0):
    raise ValueError("`x` must be strictly increasing sequence.")
  y = jnp.moveaxis(y, axis, 0)
  if dydx is not None:
    dydx = jnp.moveaxis(dydx, axis, 0)
  return x, dx, y, axis, dydx
