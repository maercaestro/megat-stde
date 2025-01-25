import torch
import torch.nn as nn
import torch.optim as optim
from src.stde import STDE
from src.models import MLP


#----------------------------------------
#1. Poisson 1D
#----------------------------------------

def test_poisson_1d():
    def f_exact(x):
        return torch.zeros_like(x)  # Source term f(x) = 0

    def u_exact(x):
        return x**2

    # Define STDE operator for second derivative
    operator_coeffs = {(0, 0): 1.0}
    stde = STDE(input_dim=1, order=2, operator_coefficients=operator_coeffs)

    # Create model and optimizer
    model = MLP(input_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Define collocation points
    x_interior = torch.linspace(0, 1, 20, requires_grad=True).view(-1, 1)
    x_boundary = torch.tensor([[0.0], [1.0]])

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()

        # PDE residuals
        pde_residuals = []
        for xi in x_interior:
            xi_ = xi.unsqueeze(0)  # Shape [1, 1]
            d2u_approx = stde.apply_operator(lambda x: model(x), [xi_], num_samples=10)
            pde_residuals.append((d2u_approx - f_exact(xi_))**2)
        pde_loss = torch.mean(torch.stack(pde_residuals))

        # Boundary conditions
        u_left, u_right = model(x_boundary[0]), model(x_boundary[1])
        bc_loss = u_left**2 + (u_right - u_exact(x_boundary[1]))**2

        # Total loss
        loss = pde_loss + bc_loss
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[Poisson 1D] Epoch {epoch}, Loss: {loss.item()}")

    # Test solution
    x_test = torch.linspace(0, 1, 10).view(-1, 1)
    u_pred = model(x_test).detach()
    u_true = u_exact(x_test)
    print("[Poisson 1D] Final MSE:", torch.mean((u_pred - u_true)**2).item())

#----------------------------------------
#2. Poisson 2D
#----------------------------------------

def test_poisson_2d():
    def f_exact(xy):
        return torch.zeros_like(xy[:, 0])  # Source term f(x, y) = 0

    def u_exact(xy):
        return xy[:, 0] * xy[:, 1]

    # Define STDE operator for Laplacian
    operator_coeffs = {(0, 0): 1.0, (1, 1): 1.0}  # Second derivatives w.r.t. x and y
    stde = STDE(input_dim=2, order=2, operator_coefficients=operator_coeffs)

    # Create model and optimizer
    model = MLP(input_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Define collocation points
    xy_interior = torch.rand((100, 2), requires_grad=True)  # Shape [100, 2]
    xy_boundary = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=False)

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()

        # PDE residuals
        pde_residuals = []
        for xy in xy_interior:
            xy_ = xy.unsqueeze(0)  # Shape [1, 2]

            # Split xy_ into separate tensors for x and y
            x, y = xy_[:, 0:1], xy_[:, 1:2]

            # Pass split tensors to STDE
            laplace_u = stde.apply_operator(lambda x, y: model(torch.cat([x, y], dim=1)), [x, y], num_samples=10)
            pde_residuals.append((laplace_u - f_exact(xy_))**2)
        pde_loss = torch.mean(torch.stack(pde_residuals))

        # Boundary conditions
        u_boundary = model(xy_boundary)
        bc_loss = torch.mean((u_boundary - u_exact(xy_boundary))**2)

        # Total loss
        loss = pde_loss + bc_loss
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[Poisson 2D] Epoch {epoch}, Loss: {loss.item()}")

    # Test solution
    xy_test = torch.rand((10, 2), requires_grad=False)  # Random test points
    u_pred = model(xy_test).detach()
    u_true = u_exact(xy_test)
    print("[Poisson 2D] Final MSE:", torch.mean((u_pred - u_true)**2).item())

#_______________________________________________________________________________________
#3. Heat Equation 1D
#_______________________________________________________________________________________
def test_heat_equation():
    def f_exact(x):
        return torch.exp(-torch.pi**2 * torch.tensor(0.01)) * torch.sin(torch.pi * x)

    def u_exact(x):
        return f_exact(x)

    # Define STDE operator for time and space derivatives
    operator_coeffs = {(0, 0): 1.0}  # Second derivative w.r.t. x
    stde = STDE(input_dim=1, order=2, operator_coefficients=operator_coeffs)

    # Create model and optimizer
    model = MLP(input_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Define collocation points
    x_interior = torch.linspace(0, 1, 20, requires_grad=True).view(-1, 1)
    x_boundary = torch.tensor([[0.0], [1.0]])

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()

        # PDE residuals
        pde_residuals = []
        for xi in x_interior:
            xi_ = xi.unsqueeze(0)  # Shape [1, 1]
            d2u_approx = stde.apply_operator(lambda x: model(x), [xi_], num_samples=10)
            pde_residuals.append((d2u_approx - f_exact(xi_))**2)
        pde_loss = torch.mean(torch.stack(pde_residuals))

        # Boundary loss
        u_left, u_right = model(x_boundary[0]), model(x_boundary[1])
        bc_loss = u_left**2 + (u_right - u_exact(x_boundary[1]))**2

        # Total loss
        loss = pde_loss + bc_loss
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[Heat Equation] Epoch {epoch}, Loss: {loss.item()}")

    # Test solution
    x_test = torch.linspace(0, 1, 10).view(-1, 1)
    u_pred = model(x_test).detach()
    u_true = u_exact(x_test)
    print("[Heat Equation] Final MSE:", torch.mean((u_pred - u_true)**2).item())

# ----------------------------------------
# Run All Tests
# ----------------------------------------

if __name__ == "__main__":
    print("Testing 1D Poisson Equation...")
    test_poisson_1d()

    print("\nTesting 2D Poisson Equation...")
    test_poisson_2d()

    print("\nTesting Heat Equation...")
    test_heat_equation()