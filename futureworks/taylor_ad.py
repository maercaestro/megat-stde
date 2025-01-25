import torch
import math
import time

class TaylorModeAD:
    def __init__(self, func, expansion_point, max_order=2):
        """
        Taylor-mode Automatic Differentiation for higher-order derivatives.

        Args:
            func (callable): Function for which derivatives are computed.
            expansion_point (float or torch.Tensor): Point around which Taylor expansion is performed.
            max_order (int): Maximum order of derivatives to compute.
        """
        self.func = func
        self.expansion_point = expansion_point
        self.max_order = max_order
        self.expansion_value = None
        self.derivatives = []  # List to store derivatives up to max_order
        self._compute_derivatives()

    def _compute_derivatives(self):
        """
        Compute derivatives up to the specified order at the expansion point.
        """
        expansion_point_tensor = torch.tensor(
            [self.expansion_point], requires_grad=True, dtype=torch.float32
        )
        current_value = self.func(expansion_point_tensor)
        self.expansion_value = current_value.item()
        self.derivatives = []

        # Compute derivatives iteratively
        for order in range(1, self.max_order + 1):
            grad = torch.autograd.grad(
                current_value,
                expansion_point_tensor,
                create_graph=True,
                retain_graph=True,
            )[0]
            self.derivatives.append(grad.item())
            current_value = grad  # Use the gradient as the value for the next derivative

    def approximate(self, inputs):
        """
        Approximate the function value for given inputs using the Taylor series.
    
        Args:
            inputs (torch.Tensor): Input tensor.
    
        Returns:
            torch.Tensor: Approximated function values.
        """
        inputs = inputs.clone().detach()  # Avoid reconstructing tensors
        result = torch.full_like(inputs, self.expansion_value)  # Initialize with expansion value
    
        # Apply Taylor expansion: f(x) ≈ f(a) + f'(a)(x-a) + f''(a)/2!(x-a)^2 + ...
        for order, derivative in enumerate(self.derivatives, start=1):
            factorial = torch.exp(torch.lgamma(torch.tensor(order + 1, dtype=torch.float32)))
            result += derivative * ((inputs - self.expansion_point) ** order) / factorial
    
        return result

