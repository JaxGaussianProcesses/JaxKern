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

from typing import Callable, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxutils import Parameters
from .base import AbstractKernelComputation


class EigenKernelComputation(AbstractKernelComputation):
    def __init__(
        self,
        kernel_fn: Callable[
            [Parameters, Float[Array, "1 D"], Float[Array, "1 D"]], Array
        ] = None,
    ) -> None:
        super().__init__(kernel_fn)
        self._eigenvalues = None
        self._eigenvectors = None
        self._num_verticies = None

    # Define an eigenvalue setter and getter property
    @property
    def eigensystem(self) -> Tuple[Float[Array, "N"], Float[Array, "N N"], int]:
        """Returns the eigenvalues, eigenvectorsof the graph Laplacian and
        number of vertices in the graph.

        Returns:
            Tuple[Float[Array], Float[Array], int]: The eigenvalues, eigenvectors,
                and number of vertices in the graph.
        """
        return self._eigenvalues, self._eigenvectors, self._num_verticies

    @eigensystem.setter
    def eigensystem(
        self, eigenvalues: Float[Array, "N"], eigenvectors: Float[Array, "N N"]
    ) -> None:
        """Set the eigenvalues and eigenvectors of the graph Laplacian.

        Args:
            eigenvalues (Float[Array]): The eigenvalues of the graph Laplacian.
            eigenvectors (Float[Array]): The eigenvectors of the graph Laplacian.
        """
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

    @property
    def num_vertex(self) -> int:
        """Returns the number of vertices in the graph.

        Returns:
            int: The number of vertices in the graph.
        """
        return self._num_verticies

    @num_vertex.setter
    def num_vertex(self, num_vertex: int) -> None:
        self._num_verticies = num_vertex

    def _compute_S(self, params: Parameters) -> Float[Array, "N"]:
        """Transform the eigenvalues of the graph Laplacian according to the
        RBF kernel's SPDE form.

        Args:
            params (Parameters): The parameters used to transform the Laplacian.
                This will contain a smoothness, lengthscale, and variance parameter

        Returns:
            Float[Array, "N"]: The transformed eigenvalues.
        """
        evals, _ = self.eigensystem
        S = jnp.power(
            evals
            + 2 * params["smoothness"] / params["lengthscale"] / params["lengthscale"],
            -params["smoothness"],
        )
        S = jnp.multiply(S, self.num_vertex / jnp.sum(S))
        S = jnp.multiply(S, params["variance"])
        return S

    def cross_covariance(
        self, params: Parameters, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        """Compute the cross covariance matrix between two sets of points.

        Args:
            params (Parameters): The parameters used to compute the covariance matrix.
            x (Float[Array, "N D"]): The first set of points.
            y (Float[Array, "M D"]): The second set of points.

        Returns:
            Float[Array, "N M"]: The NxM cross covariance matrix.
        """
        S = self._compute_S(params=params)
        matrix = self.kernel_fn(params, x, y, S=S)
        return matrix
