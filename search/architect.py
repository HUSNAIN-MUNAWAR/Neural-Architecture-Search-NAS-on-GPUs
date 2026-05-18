"""Architecture optimizer for first-order and unrolled DARTS updates."""

from __future__ import annotations

import copy
import logging
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

from utils.gpu_utils import unwrap_model


LOGGER = logging.getLogger(__name__)


class Architect:
    """Handle optimization of architecture parameters alpha."""

    def __init__(
        self,
        model: nn.Module,
        arch_learning_rate: float,
        arch_weight_decay: float,
    ) -> None:
        self.model = unwrap_model(model)
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
        )

    def step(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
        eta: float,
        network_optimizer: torch.optim.Optimizer,
        unrolled: bool,
    ) -> torch.Tensor:
        """Update architecture parameters using validation loss."""
        self.optimizer.zero_grad(set_to_none=True)
        if unrolled:
            loss = self._backward_step_unrolled(
                input_train,
                target_train,
                input_valid,
                target_valid,
                eta,
                network_optimizer,
            )
        else:
            loss = self._backward_step(input_valid, target_valid)
        self.optimizer.step()
        return loss.detach()

    def _backward_step(self, input_valid: torch.Tensor, target_valid: torch.Tensor) -> torch.Tensor:
        """First-order DARTS architecture update."""
        loss = self.model.loss(input_valid, target_valid)
        loss.backward()
        return loss

    def _concat(self, xs: Iterable[torch.Tensor]) -> torch.Tensor:
        """Flatten and concatenate parameter tensors."""
        return torch.cat([x.view(-1) for x in xs])

    def _construct_model_from_theta(self, theta: torch.Tensor) -> nn.Module:
        """Create a copied model whose weight parameters are set from a flat tensor."""
        model_new = copy.deepcopy(self.model)
        offset = 0
        for param in model_new.weight_parameters():
            numel = param.numel()
            param.data.copy_(theta[offset : offset + numel].view_as(param))
            offset += numel
        return model_new.to(next(self.model.parameters()).device)

    def _compute_unrolled_model(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        eta: float,
        network_optimizer: torch.optim.Optimizer,
    ) -> nn.Module:
        """Compute one virtual SGD step on network weights."""
        loss = self.model.loss(input_train, target_train)
        theta = self._concat(self.model.weight_parameters()).data
        try:
            moment = self._concat(
                network_optimizer.state[v].get("momentum_buffer", torch.zeros_like(v))
                for v in self.model.weight_parameters()
            ).mul_(network_optimizer.param_groups[0].get("momentum", 0.0))
        except Exception:
            moment = torch.zeros_like(theta)
        grads = torch.autograd.grad(loss, self.model.weight_parameters(), create_graph=True)
        dtheta = self._concat(grads).data + network_optimizer.param_groups[0].get("weight_decay", 0.0) * theta
        unrolled_theta = theta.sub(eta, moment + dtheta)
        return self._construct_model_from_theta(unrolled_theta)

    def _backward_step_unrolled(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
        eta: float,
        network_optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """Second-order DARTS approximation using finite-difference Hessian-vector product."""
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model.loss(input_valid, target_valid)
        unrolled_arch_params = unrolled_model.arch_parameters()
        unrolled_weight_params = unrolled_model.weight_parameters()
        dalpha = torch.autograd.grad(unrolled_loss, unrolled_arch_params, retain_graph=True)
        vector = torch.autograd.grad(unrolled_loss, unrolled_weight_params)
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        for grad, implicit_grad in zip(dalpha, implicit_grads):
            grad.data.sub_(eta, implicit_grad.data)
        for param, grad in zip(self.model.arch_parameters(), dalpha):
            if param.grad is None:
                param.grad = grad.detach().clone()
            else:
                param.grad.data.copy_(grad.data)
        return unrolled_loss

    def _hessian_vector_product(
        self,
        vector: Sequence[torch.Tensor],
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        r: float = 1e-2,
    ) -> List[torch.Tensor]:
        """Estimate Hessian-vector product for architecture parameters."""
        norm = torch.cat([v.view(-1) for v in vector]).norm()
        eps = r / (norm + 1e-12)
        for param, v in zip(self.model.weight_parameters(), vector):
            param.data.add_(eps, v)
        loss = self.model.loss(input_train, target_train)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
        for param, v in zip(self.model.weight_parameters(), vector):
            param.data.sub_(2.0 * eps, v)
        loss = self.model.loss(input_train, target_train)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())
        for param, v in zip(self.model.weight_parameters(), vector):
            param.data.add_(eps, v)
        return [(p - n).div_(2.0 * eps) for p, n in zip(grads_p, grads_n)]
