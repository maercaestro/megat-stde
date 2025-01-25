# --------------------------------
# This script compares the training time of STDE vs. normal autograd for solving a simple 1D PDE
# --------------------------------


import torch
import torch.nn as nn
import torch.optim as optim
import random,time
from src.stde import STDE
from src.models import MLP1D


#----------------------------------------
#1. Poisson 1D using STDE
#----------------------------------------
def pinn_1d_demo_stde():
    # PDE: u''(x) = 2
    # BC:  u(0)=0,  u(1)=1
    # This uses STDE to approximate second derivative

    torch.manual_seed(0)
    random.seed(0)
    
    # 1a. Create your STDE for second derivative
    operator_coeffs = {
        (0, 0): 1.0  # second derivative wrt x
    }
    stde = STDE(
        input_dim=1,
        order=2,
        operator_coefficients=operator_coeffs,
        rand_batch_size=0,
        use_abs_for_sampling=True
    )

    # 1b. Simple MLP
    model = MLP1D(hidden_size=16)

    # 1c. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 1d. Collocation points in [0,1]
    n_interior = 20
    x_interior = torch.linspace(0.0, 1.0, n_interior, requires_grad=True).view(-1, 1)
    x_left = torch.tensor([[0.0]])
    x_right = torch.tensor([[1.0]])

    # 1e. Training loop
    epochs = 2000

    start_time = time.time()  # Start timing

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # PDE residual: d2u - 2 = 0
        pde_res_terms = []
        for xi in x_interior:
            xi_ = xi.unsqueeze(0)  # shape [1,1]
            # approximate second derivative using STDE
            d2u_approx = stde.apply_operator(lambda x: model(x), [xi_], num_samples=5)
            pde_res = d2u_approx - 2.0
            pde_res_terms.append(pde_res**2)

        pde_loss = torch.mean(torch.stack(pde_res_terms))

        # Boundary conditions: u(0)=0, u(1)=1
        u_left = model(x_left)
        u_right = model(x_right)
        bc_loss = (u_left**2 + (u_right - 1.0)**2).mean()

        # Total loss
        loss = pde_loss + bc_loss

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"[STDE] Epoch {epoch:4d}, Loss: {loss.item():.6f}, PDE: {pde_loss.item():.6f}, BC: {bc_loss.item():.6f}")

    elapsed_stde = time.time() - start_time
    print(f"\n[STDE] Training time: {elapsed_stde:.4f} seconds\n")

    # Test the final result
    x_test = torch.linspace(0, 1, 11).view(-1, 1)
    with torch.no_grad():
        u_pred = model(x_test).view(-1)
        u_true = x_test.view(-1)**2
    mse_error = torch.mean((u_pred - u_true)**2).item()

    print("[STDE] Final MSE on 11 test points:", mse_error)
    return elapsed_stde, mse_error

#----------------------------------------
#2. Poisson 1D using normal autograd
#----------------------------------------


def pinn_1d_demo_autograd():
    # PDE: u''(x) = 2
    # BC:  u(0)=0,  u(1)=1
    # This uses normal PyTorch autograd to compute second derivative

    torch.manual_seed(0)
    random.seed(0)
    
    # 2a. Simple MLP
    model = MLP1D(hidden_size=16)

    # 2b. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 2c. Collocation points in [0,1]
    n_interior = 20
    x_interior = torch.linspace(0.0, 1.0, n_interior, requires_grad=True).view(-1, 1)
    x_left = torch.tensor([[0.0]], requires_grad=False)
    x_right = torch.tensor([[1.0]], requires_grad=False)

    # 2d. Training loop
    epochs = 2000

    start_time = time.time()  # Start timing

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # PDE residual: d2u - 2 = 0
        pde_res_terms = []
        for xi in x_interior:
            xi_ = xi.unsqueeze(0)  # shape [1,1]
            u = model(xi_)  # shape [1,1]
            # first derivative
            grad_u = torch.autograd.grad(u, xi_, create_graph=True)[0]    # shape [1,1]
            # second derivative
            grad2_u = torch.autograd.grad(grad_u, xi_, create_graph=True)[0]  # shape [1,1]
            pde_res = grad2_u - 2.0
            pde_res_terms.append(pde_res**2)

        pde_loss = torch.mean(torch.stack(pde_res_terms))

        # Boundary conditions: u(0)=0, u(1)=1
        u_left = model(x_left)
        u_right = model(x_right)
        bc_loss = (u_left**2 + (u_right - 1.0)**2).mean()

        loss = pde_loss + bc_loss

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"[Autograd] Epoch {epoch:4d}, Loss: {loss.item():.6f}, PDE: {pde_loss.item():.6f}, BC: {bc_loss.item():.6f}")

    elapsed_auto = time.time() - start_time
    print(f"\n[Autograd] Training time: {elapsed_auto:.4f} seconds\n")

    # Test the final result
    x_test = torch.linspace(0, 1, 11).view(-1, 1)
    with torch.no_grad():
        u_pred = model(x_test).view(-1)
        u_true = x_test.view(-1)**2
    mse_error = torch.mean((u_pred - u_true)**2).item()

    print("[Autograd] Final MSE on 11 test points:", mse_error)
    return elapsed_auto, mse_error

#----------------------------------------
# Compare the time for both method
#----------------------------------------

if __name__ == "__main__":
    time_stde, mse_stde = pinn_1d_demo_stde()
    time_auto, mse_auto = pinn_1d_demo_autograd()

    print("-" * 40)
    print(f"STDE training time:      {time_stde:.4f} s, MSE: {mse_stde:.6f}")
    print(f"Autograd training time:  {time_auto:.4f} s, MSE: {mse_auto:.6f}")
    print("-" * 40)