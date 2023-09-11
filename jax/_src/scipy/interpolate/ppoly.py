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

import functools
import typing

import scipy.interpolate

import jax
import jax.numpy as jnp
from jax._src.numpy.util import _wraps


class _PPolyBase(typing.NamedTuple):
  c: jax.Array
  x: jax.Array
  extrapolate: bool
  axis: int

  @classmethod
  def init(cls, c: jax.Array, x: jax.Array, extrapolate: typing.Optional[bool] = None, axis: int = 0):
    if extrapolate is None:
      extrapolate = True
    elif extrapolate != 'periodic':
      extrapolate = bool(extrapolate)
    if c.ndim < 2:
      raise ValueError("Coefficients array must be at least 2-dimensional.")
    if not (0 <= axis < c.ndim - 1):
      raise ValueError(f"axis={axis} must be between 0 and {c.ndim-1}")
    if axis != 0:
      c = jnp.moveaxis(c, axis+1, 0)
      c = jnp.moveaxis(c, axis+1, 0)
    if x.ndim != 1:
      raise ValueError("x must be 1-dimensional")
    if x.size < 2:
      raise ValueError("at least 2 breakpoints are needed")
    if c.ndim < 2:
      raise ValueError("c must have at least 2 dimensions")
    if c.shape[0] == 0:
      raise ValueError("polynomial must be at least of order 0")
    if c.shape[1] != x.size-1:
      raise ValueError("number of coefficients != len(x)-1")
    dx = jnp.diff(x)
    if not (jnp.all(dx >= 0) or jnp.all(dx <= 0)):
      raise ValueError("`x` must be strictly increasing or decreasing.")
    return cls(
      c=c,
      x=x,
      extrapolate=extrapolate,
      axis=axis)

  def extend(self, c: jax.Array, x):
    raise NotImplementedError

  def __call__(self, x: jax.Array, nu: int = 0, extrapolate: typing.Optional[bool] = None):
    """Evaluate the piecewise polynomial or its derivative."""
    if extrapolate is None:
      extrapolate = self.extrapolate
    if extrapolate == 'periodic':
      x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
      extrapolate = False
    c = self.c.reshape(self.c.shape[0], self.c.shape[1], -1)
    out = _evaluate(c, self.x, x, nu, extrapolate)
    out = out.reshape(x.shape + self.c.shape[2:])
    if self.axis != 0:
      l = list(range(out.ndim))
      l = l[x.ndim:x.ndim+self.axis] + l[:x.ndim] + l[x.ndim+self.axis:]
      out = out.transpose(l)
    return out


@_wraps(scipy.interpolate.PPoly)
class PPoly(_PPolyBase):
  """Piecewise polynomial in terms of coefficients and breakpoints."""

  def derivative(self, nu: int = 1):
    """Construct a new piecewise polynomial representing the derivative."""
    raise NotImplementedError

  def antiderivative(self, nu=1):
    """Construct a new piecewise polynomial representing the antiderivative."""
    raise NotImplementedError

  def integrate(self, a, b, extrapolate=None):
    """Compute a definite integral over a piecewise polynomial."""
    raise NotImplementedError

  def solve(self, y: float = 0., discontinuity: bool = True, extrapolate: typing.Optional[bool] = None):
    """Find real solutions of the equation ``pp(x) == y``."""
    raise NotImplementedError

  def roots(self, discontinuity: bool = True, extrapolate: typing.Optional[bool] = None):
    """Find real roots of the piecewise polynomial."""
    raise NotImplementedError

  @classmethod
  def from_spline(cls, tck, extrapolate=None):
    """Construct a piecewise polynomial from a spline."""
    raise NotImplementedError

  @classmethod
  def from_bernstein_basis(cls, bp, extrapolate=None):
    """Construct a polynomial in the power basis from the Bernstein basis."""
    raise NotImplementedError


@functools.partial(jnp.vectorize, signature='(k,m,n),(m1),(),(),()->(n)')
def _evaluate(c: jax.Array,
              x: jax.Array,
              xval: jax.Array,
              dx: int,
              extrapolate: bool,
              ) -> jax.Array:
  """Evaluate a piecewise polynomial."""
  if dx < 0:
    raise ValueError("Order of derivative cannot be negative")
  if c.shape[1] != x.shape[0] - 1:
    raise ValueError("x and c have incompatible shapes")
  ascending = x[x.shape[0] - 1] >= x[0]
  i_ascending = _find_interval_ascending(x[0], x.shape[0], xval, extrapolate)
  i_descending = _find_interval_descending(x[0], x.shape[0], xval, extrapolate)
  i = jnp.where(ascending, i_ascending, i_descending)
  interval = jnp.where(i < 0, 0, i)
  out = _evaluate_poly1(
    s=xval - x[interval],
    c=c,
    ci=interval,
    cj=jnp.arange(c.shape[2]),
    dx=dx)
  return out


@functools.partial(jnp.vectorize, signature='(),(k,m,n),(),(),()->()')
def _evaluate_poly1(s: jax.Array, c: jax.Array, ci: int, dx: int) -> jax.Array:
  """Evaluate polynomial, derivative, or antiderivative in a single interval."""
  res = 0.0
  k = jnp.arange(c.shape[0])
  if dx == 0:
    prefactor = jnp.ones(c.shape[0], s.dtype)
    z = jnp.cumprod(s)
  elif dx > 0:
    raise NotImplementedError
  else:
    raise NotImplementedError
  res = jnp.sum(c[c.shape[0] - k - 1, ci] * z * prefactor)
  return res


@functools.partial(jnp.vectorize, signature='(m1),(),(),(),()->()')
def _find_interval_ascending(x: jax.Array,
                             nx: int,
                             xval: jax.Array,
                             extrapolate: bool = True,
                             ) -> int:
  """Find an interval such that x[interval] <= xval < x[interval+1]."""
  in_bounds = jnp.logical_and(x[0] <= xval, xval <= x[nx - 1])
  interval = jnp.searchsorted(x, xval, side='right')
  interval = jnp.clip(interval, 0, nx - 2)
  interval = jnp.where(jnp.logical_or(in_bounds, extrapolate), interval, -1)
  return interval


@functools.partial(jnp.vectorize, signature='(m1),(),(),(),()->()')
def _find_interval_descending(x: jax.Array,
                              nx: int,
                              xval: jax.Array,
                              extrapolate: bool = True,
                              ) -> int:
  """Find an interval such that x[interval + 1] < xval <= x[interval]."""
  in_bounds = jnp.logical_and(x[nx - 1] <= xval, xval <= x[0])
  interval = x.size - jnp.searchsorted(x, xval, side='right', sorter=jnp.arange(x.size)[::-1])
  interval = jnp.clip(interval, 0, nx - 2)
  interval = jnp.where(jnp.logical_or(in_bounds, extrapolate), interval, -1)
  return interval
