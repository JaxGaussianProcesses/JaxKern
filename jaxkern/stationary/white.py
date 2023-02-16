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

from typing import Dict, Optional, List

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..base import AbstractKernel
from ..computations import (
    ConstantDiagonalKernelComputation,
)
from jaxutils import param
from jaxutils.bijectors import Softplus


class White(AbstractKernel):
    variance: Float[Array, "1"] = param(Softplus)

    def __init__(
        self,
        variance: Float[Array, "1"] = jnp.array([1.0]),
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "White Noise Kernel",
    ) -> None:
        super().__init__(
            ConstantDiagonalKernelComputation,
            active_dims,
            spectral_density=None,
            name=name,
        )
        self._stationary = True
        self.variance = variance

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\delta(x-y)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        K = jnp.all(jnp.equal(x, y)) * params["variance"]
        return K.squeeze()
