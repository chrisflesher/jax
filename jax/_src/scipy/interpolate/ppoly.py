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

import scipy.spatial.interpolate

import jax
import jax.numpy as jnp
import numpy as onp
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

  def __call__(self, x, nu: int = 0, extrapolate: typing.Optional[bool] = None):
    """Evaluate the piecewise polynomial or its derivative."""
    if extrapolate is None:
      extrapolate = self.extrapolate
    if extrapolate == 'periodic':
      x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
      extrapolate = False
    out = _evaluate(
      c=self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
      x=self.x,
      xp=x,
      dx=nu,
      extrapolate=bool(extrapolate))
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
    if extrapolate is None:
      extrapolate = self.extrapolate
    if jnp.issubdtype(self.c.dtype, jnp.complexfloating):
      raise ValueError("Root finding is only for real-valued polynomials")
    r = _real_roots(
      c=self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
      x=self.x,
      y=y,
      report_discont=bool(discontinuity),
      extrapolate=bool(extrapolate))
    if self.c.ndim == 2:
      return r[0]
    else:
      r2 = jnp.empty(onp.prod(self.c.shape[2:]), dtype=object)
      # this for-loop is equivalent to ``r2[...] = r``, but that's broken
      # in NumPy 1.6.0
      for ii, root in enumerate(r):
        r2[ii] = root
      return r2.reshape(self.c.shape[2:])

  def roots(self, discontinuity: bool = True, extrapolate: typing.Optional[bool] = None):
    """Find real roots of the piecewise polynomial."""
    return self.solve(
      y=0.,
      discontinuity=discontinuity,
      extrapolate=extrapolate)

  @classmethod
  def from_spline(cls, tck, extrapolate=None):
    """Construct a piecewise polynomial from a spline."""
    raise NotImplementedError

  @classmethod
  def from_bernstein_basis(cls, bp, extrapolate=None):
    """Construct a polynomial in the power basis from the Bernstein basis."""
    raise NotImplementedError


@functools.partial(jnp.vectorize, signature='(k,m,n),(),(),(k),(k)->(r,n)')
def _croots_poly1(c: jax.Array, y: float, ci: int, cj: int, wr: jax.Array, wi: jax.Array) -> jax.Array:
  """Find all complex roots of a local polynomial."""
  n = c.shape[0]

  # Check actual polynomial order
  for j in range(n):
    if c[j,ci,cj] != 0:
      order = n - 1 - j
      break
  else:
    order = -1

  if order < 0:
    # Zero everywhere
    if y == 0:
      return -1
    else:
      return 0
  elif order == 0:
    # Nonzero constant polynomial: no roots
    # (unless r.h.s. is exactly equal to the coefficient, that is.)
    if c[n-1, ci, cj] == y:
      return -1
    else:
      return 0
  elif order == 1:
    # Low-order polynomial: a0*x + a1
    a0 = c[n-1-order,ci,cj]
    a1 = c[n-1-order+1,ci,cj] - y
    wr[0] = -a1 / a0
    wi[0] = 0
    return 1
  elif order == 2:
    # Low-order polynomial: a0*x**2 + a1*x + a2
    a0 = c[n-1-order,ci,cj]
    a1 = c[n-1-order+1,ci,cj]
    a2 = c[n-1-order+2,ci,cj] - y

    d = a1*a1 - 4*a0*a2
    if d < 0:
      # no real roots
      d = libc.math.sqrt(-d)
      wr[0] = -a1/(2*a0)
      wi[0] = -d/(2*a0)
      wr[1] = -a1/(2*a0)
      wi[1] = d/(2*a0)
      return 2

    d = libc.math.sqrt(d)

    # avoid cancellation in subtractions
    if d == 0:
      wr[0] = -a1/(2*a0)
      wi[0] = 0
      wr[1] = -a1/(2*a0)
      wi[1] = 0
    elif a1 < 0:
      wr[0] = (2*a2) / (-a1 + d) # == (-a1 - d)/(2*a0)
      wi[0] = 0
      wr[1] = (-a1 + d) / (2*a0)
      wi[1] = 0
    else:
      wr[0] = (-a1 - d)/(2*a0)
      wi[0] = 0
      wr[1] = (2*a2) / (-a1 - d) # == (-a1 + d)/(2*a0)
      wi[1] = 0

    return 2

  # Compute required workspace and allocate it
  lwork = 1 + 8*n

  if workspace[0] == NULL:
    nworkspace = n*n + lwork
    workspace[0] = libc.stdlib.malloc(nworkspace * sizeof(double))
    if workspace[0] == NULL:
      raise MemoryError("Failed to allocate memory in croots_poly1")

  a = <double*>workspace[0]
  work = a + n*n

  # Initialize the companion matrix, Fortran order
  for j in range(order*order):
    a[j] = 0
  for j in range(order):
    cc = c[n-1-j,ci,cj]
    if j == 0:
      cc -= y
    a[j + (order-1)*order] = -cc / c[n-1-order,ci,cj]
    if j + 1 < order:
      a[j+1 + order*j] = 1

  # Compute companion matrix eigenvalues
  info = 0
  dgeev("N", "N", &order, a, &order, <double*>wr, <double*>wi,
      NULL, &order, NULL, &order, work, &lwork, &info)
  if info != 0:
    # Failure
    return -2

  # Sort roots (insertion sort)
  for i in range(order):
    br = wr[i]
    bi = wi[i]
    for j in range(i - 1, -1, -1):
      if wr[j] > br:
        wr[j+1] = wr[j]
        wi[j+1] = wi[j]
      else:
        wr[j+1] = br
        wi[j+1] = bi
        break
    else:
      wr[0] = br
      wi[0] = bi

  # Return with roots
  return order


@functools.partial(jnp.vectorize, signature='(k,m,n),(m+1),(),(),()->(n)')
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
def _evaluate_poly1(s: jax.Array, c: jax.Array, ci: int, cj: int, dx: int) -> jax.Array:
  """Evaluate polynomial, derivative, or antiderivative in a single interval."""
  res = 0.0
  z = jnp.where(dx >= 0, 1.0, z * s**(-dx))
  for kp in range(c.shape[0]):
    # prefactor of term after differentiation
    if dx == 0:
      prefactor = 1.0
    elif dx > 0:
      # derivative
      if kp < dx:
        continue
      else:
        prefactor = 1.0
        for k in range(kp, kp - dx, -1):
          prefactor *= k
    else:
      prefactor = 1.0
      for k in range(kp, kp - dx):
        prefactor /= k + 1
    res = res + c[c.shape[0] - kp - 1, ci, cj] * z * prefactor
    z = jnp.where(jnp.logical_and(kp < c.shape[0] - 1, kp >= dx), z * s, z)
  return res


@functools.partial(jnp.vectorize, signature='(m),(),(),(),()->()')
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


@functools.partial(jnp.vectorize, signature='(m),(),(),(),()->()')
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


@functools.partial(jnp.vectorize, signature='(k,m,n),(m+1),(),(),()->()')
def _real_roots(c: jax.Array, x: jax.Array, y: float, report_discont: bool, extrapolate: bool) -> jax.Array:
  """Compute real roots of a real-valued piecewise polynomial function."""
  if c.shape[1] != x.shape[0] - 1:
    raise ValueError("x and c have incompatible shapes")
  if c.shape[0] == 0:
    return jnp.array([], dtype=float)
  wr = <double*>libc.stdlib.malloc(c.shape[0] * sizeof(double))
  wi = <double*>libc.stdlib.malloc(c.shape[0] * sizeof(double))
  if not wr or not wi:
    libc.stdlib.free(wr)
    libc.stdlib.free(wi)
    raise MemoryError("Failed to allocate memory in real_roots")
  workspace = None
  last_root = jnp.nan
  ascending = x[x.shape[0] - 1] >= x[0]
  roots = []
  for jp in range(c.shape[2]):
    cur_roots = []
    for interval in range(c.shape[1]):
      # Check for sign change across intervals
      if interval > 0 and report_discont:
        va = _evaluate_poly1(x=x[interval] - x[interval-1], c=c, ci=interval-1, cj=jp, dx=0) - y
        vb = _evaluate_poly1(x=0, c=c, ci=interval, cj=jp, dx=0) - y
        if (va < 0 and vb > 0) or (va > 0 and vb < 0):
          # sign change between intervals
          if x[interval] != last_root:
            last_root = x[interval]
            cur_roots.append(float(last_root))

      # Compute first the complex roots
      k = _croots_poly1(c, y, interval, jp, wr, wi, &workspace)

      # Check for errors and identically zero values
      if k == -1:
        # Zero everywhere
        if x[interval] == x[interval+1]:
          # Only a point
          if x[interval] != last_root:
            last_root = x[interval]
            cur_roots.append(x[interval])
        else:
          # A real interval
          cur_roots.append(x[interval])
          cur_roots.append(np.nan)
          last_root = libc.math.NAN
        continue
      elif k < -1:
        # An error occurred
        raise RuntimeError("Internal error in root finding; please report this bug")
      elif k == 0:
        # No roots
        continue

      # Filter real roots
      for i in range(k):
        # Check real root
        #
        # The reality of a root is a decision that can be left to LAPACK,
        # which has to determine this in any case.
        if wi[i] != 0:
          continue

        # Refine root by one Newton iteration
        f = _evaluate_poly1(wr[i], c, interval, jp, 0) - y
        df = _evaluate_poly1(wr[i], c, interval, jp, 1)
        if df != 0:
          dx = f/df
          if abs(dx) < abs(wr[i]):
            wr[i] = wr[i] - dx

        # Check interval
        wr[i] += x[interval]
        if interval == 0 and extrapolate:
          # Half-open to the left/right.
          # Might also be the only interval, in which case there is
          # no limitation.
          if (interval != c.shape[1] - 1 and
            (ascending and not wr[i] <= x[interval+1] or
             not ascending and not wr[i] >= x[interval + 1])):
              continue
        elif interval == c.shape[1] - 1 and extrapolate:
          # Half-open to the right/left.
          if (ascending and not wr[i] >= x[interval] or
            not ascending and not wr[i] <= x[interval]):
              continue
        else:
          if (ascending and
            not x[interval] <= wr[i] <= x[interval+1] or
            not ascending and
            not x[interval + 1] <= wr[i] <= x[interval]):
              continue

        # Add to list
        if wr[i] != last_root:
          last_root = wr[i]
          cur_roots.append(float(last_root))

      # Construct roots
      roots.append(np.array(cur_roots, dtype=float))

  return roots
