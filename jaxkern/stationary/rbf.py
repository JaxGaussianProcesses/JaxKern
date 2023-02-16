# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import List, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float
import distrax as dx

from ..base import AbstractKernel
from ..computations import (
    DenseKernelComputation,
)
from .utils import squared_distance

from jaxutils import param
from jaxutils.bijectors import Softplus


class RBF(AbstractKernel):
    """The Radial Basis Function (RBF) kernel."""

    lengthscale: Float[Array, "1 D"] = param(Softplus)
    variance: Float[Array, "1"] = param(Softplus)

    def __init__(
        self,
        lengthscale: Float[Array, "1 D"] = jnp.array([1.0]),
        variance: Float[Array, "1"] = jnp.array([1.0]),
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "Radial basis function kernel",
    ) -> None:
        super().__init__(DenseKernelComputation, active_dims, name)
        self._stationary = True
        self._spectral_density = dx.Normal(loc=0.0, scale=1.0)
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\\ell` and variance :math:`\\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( \\frac{\\lVert x - y \\rVert^2_2}{2 \\ell^2} \\Bigg)

        Args:
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """

        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()
