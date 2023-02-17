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

from __future__ import annotations

import abc
from typing import Callable, List, Optional, Sequence
import distrax as dx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .computations import AbstractKernelComputation, DenseKernelComputation

from jaxutils import Module
from equinox import static_field


class AbstractKernel(Module):
    """Base kernel class."""

    compute_engine: AbstractKernelComputation = static_field()
    active_dims: List[int] = static_field()
    name: str = static_field()

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "Kernel",
    ) -> None:
        self.compute_engine = compute_engine
        self.active_dims = active_dims
        self.name = name

    @property
    def ndims(self):
        return 1 if not self.active_dims else len(self.active_dims)

    @property
    def gram(self):
        return self.compute_engine(kernel_fn=self.__call__).gram

    @property
    def cross_covariance(self):
        return self.compute_engine(kernel_fn=self.__call__).cross_covariance

    @property
    def spectral_density(self) -> dx.Distribution:
        if self._spectral_density is None:
            raise NotImplementedError(
                f"The spectral density for the {self.name} kernel is not implemented."
            )

    @property
    def stationary(self) -> bool:
        """Boolean property as to whether the kernel is stationary or not.

        Returns:
            bool: True if the kernel is stationary.
        """
        return self._stationary

    # @property
    # def compute_engine(self) -> AbstractKernelComputation:
    #     """The compute engine that is used to perform the kernel computations.

    #     Returns:
    #         AbstractKernelComputation: The compute engine that is used to perform the kernel computations.
    #     """
    #     return self._compute_engine

    # @compute_engine.setter
    # def compute_engine(self, compute_engine: AbstractKernelComputation) -> None:
    #     self._compute_engine = compute_engine
    #     compute_engine = self.compute_engine(kernel_fn=self.__call__)
    #     self.gram = compute_engine.gram
    #     self.cross_covariance = compute_engine.cross_covariance

    @abc.abstractmethod
    def __call__(
        self,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        raise NotImplementedError

    def slice_input(self, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
        """Select the relevant columns of the supplied matrix to be used within the kernel's evaluation.

        Args:
            x (Float[Array, "N D"]): The matrix or vector that is to be sliced.
        Returns:
            Float[Array, "N Q"]: A sliced form of the input matrix.
        """
        return x[..., self.active_dims]

    def __add__(self, other: AbstractKernel) -> AbstractKernel:
        """Add two kernels together.
        Args:
            other (AbstractKernel): The kernel to be added to the current kernel.

        Returns:
            AbstractKernel: A new kernel that is the sum of the two kernels.
        """
        return SumKernel(kernel_set=[self, other])

    def __mul__(self, other: AbstractKernel) -> AbstractKernel:
        """Multiply two kernels together.

        Args:
            other (AbstractKernel): The kernel to be multiplied with the current kernel.

        Returns:
            AbstractKernel: A new kernel that is the product of the two kernels.
        """
        return ProductKernel(kernel_set=[self, other])

    @property
    def ard(self):
        """Boolean property as to whether the kernel is isotropic or of
        automatic relevance determination form.

        Returns:
            bool: True if the kernel is an ARD kernel.
        """
        return True if self.ndims > 1 else False


class CombinationKernel(AbstractKernel):
    """A base class for products or sums of kernels."""

    kernel_set: List[AbstractKernel] = static_field()
    combination_fn: Callable = static_field()

    def __init__(
        self,
        kernel_set: List[AbstractKernel],
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "Combination kernel",
    ) -> None:

        super().__init__(
            compute_engine=compute_engine,
            active_dims=active_dims,
            name=name,
        )

        self.kernel_set = kernel_set

        if not all(isinstance(k, AbstractKernel) for k in self.kernel_set):
            raise TypeError("can only combine Kernel instances")  # pragma: no cover
        if all(k.stationary for k in self.kernel_set):
            self._stationary = True
        self._set_kernels(self.kernel_set)

    @property
    @abc.abstractmethod
    def combination_fn(self):
        raise NotImplementedError

    def _set_kernels(self, kernels: Sequence[AbstractKernel]) -> None:
        """Combine multiple kernels. Based on GPFlow's Combination kernel."""
        # add kernels to a list, flattening out instances of this class therein
        kernels_list: List[AbstractKernel] = []
        for k in kernels:
            if isinstance(k, self.__class__):
                kernels_list.extend(k.kernel_set)
            else:
                kernels_list.append(k)

        self.kernel_set = kernels_list

    def __call__(
        self,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate combination kernel on a pair of inputs.

        Args:
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        return self.combination_fn(jnp.stack([k(x, y) for k in self.kernel_set]))


class SumKernel(CombinationKernel):
    """A kernel that is the sum of a set of kernels."""

    @property
    def combination_fn(self):
        return jnp.sum


class ProductKernel(CombinationKernel):
    """A kernel that is the product of a set of kernels."""

    @property
    def combination_fn(self):
        return jnp.prod
