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

from typing import Dict, List, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float
import distrax as dx
from ..base import AbstractKernel
from ..computations import (
    DenseKernelComputation,
)
from .utils import euclidean_distance, build_student_t_distribution
from jaxutils import param
from jaxutils.bijectors import Softplus


class Matern32(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 1.5."""

    lengthscale: Float[Array, "1 D"] = param(Softplus)
    variance: Float[Array, "1"] = param(Softplus)

    def __init__(
        self,
        lengthscale: Float[Array, "1 D"] = jnp.array([1.0]),
        variance: Float[Array, "1"] = jnp.array([1.0]),
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "Matern 3/2",
    ) -> None:
        super().__init__(DenseKernelComputation, active_dims, name)
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\\ell` and variance :math:`\\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{3}\\lvert x-y \\rvert}{\\ell^2}  \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{3}\\lvert x-y\\rvert}{\\ell^2} \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        tau = euclidean_distance(x, y)
        K = (
            params["variance"]
            * (1.0 + jnp.sqrt(3.0) * tau)
            * jnp.exp(-jnp.sqrt(3.0) * tau)
        )
        return K.squeeze()

    @property
    def spectral_density(self) -> dx.Distribution:
        return build_student_t_distribution(nu=3)

    @property
    def stationary(self) -> bool:
        return True
