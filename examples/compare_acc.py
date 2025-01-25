import torch
import torch.nn as nn
import torch.optim as optim
import random,time
from src.stde import STDE
from src.models import MLP1D



def test_stde_1d(STDE_class):
    """
    Test the STDE class on a simple 1D function f(x) = x^2,
    whose second derivative is 2 everywhere.

    Args:
        STDE_class: The STDE class to be tested.
    """

    # ----- Reproducibility -----
    torch.manual_seed(0)
    random.seed(0)

    # ----- Define the test function f(x) -----
    # f(x) = x^2
    def f(x):
        return x**2

    # ----- Define operator coefficients for the 1D second derivative -----
    # In your format, (0, 0) means "take second derivative wrt dimension 0"
    operator_coeffs = {
        (0, 0): 1.0  # coefficient = 1 for second derivative
    }

    # Create an STDE instance
    stde = STDE_class(
        input_dim=1,           # only one dimension
        order=2,               # up to second derivative
        operator_coefficients=operator_coeffs,
        rand_batch_size=0,     # use all dimensions (just 1 anyway)
        use_abs_for_sampling=True
    )

    # ----- Define input x -----
    x = torch.tensor([3.0], requires_grad=True)  # choose any value, e.g. 3.0

    # ----- Apply the operator -----
    # Increase num_samples for better accuracy if needed
    approx_value = stde.apply_operator(f, [x], num_samples=1000)

    # ----- Print results -----
    print("Approximate second derivative of x^2 at x=3.0:", approx_value.item())
    print("Exact second derivative is 2.0")
    print("Error:", abs(approx_value.item() - 2.0))


def pinn_1d_demo():
    # PDE: u''(x) = 2
    # BC:  u(0)=0,  u(1)=1
    # Exact solution: u(x) = x^2

    # Fix random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)

    # 3a. Define STDE for second derivative in 1D
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

    # 3b. Create a small MLP model
    model = MLP1D(hidden_size=16)

    # 3c. Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3d. Sample some collocation points in [0,1] for PDE
    #     (You might choose random points each epoch in a real scenario)
    n_interior = 20
    x_interior = torch.linspace(0.0, 1.0, n_interior, requires_grad=True).view(-1, 1)

    # Boundary points (no gradient needed because we'll just evaluate the model)
    x_left = torch.tensor([[0.0]])
    x_right = torch.tensor([[1.0]])

    # 3e. Training loop
    epochs = 2000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # PDE residual:
        # PDE is: u''(x) - 2 = 0
        # We'll approximate the PDE on each interior point
        pde_res_terms = []
        for xi in x_interior:
            xi_ = xi.unsqueeze(0).unsqueeze(1)  # Ensure shape [1, 1]
        
            # Apply STDE operator
            try:
                d2u_approx = stde.apply_operator(lambda x: model(x), [xi_], num_samples=5)
            except Exception as e:
                print(f"Error applying operator at xi={xi_}: {e}")
                raise

            # PDE residual = d2u_approx - 2
            pde_res = d2u_approx - 2.0
            pde_res_terms.append(pde_res**2)

        pde_loss = torch.mean(torch.stack(pde_res_terms))

        # Boundary conditions:
        #   u(0) = 0,  u(1) = 1
        u_left = model(x_left)
        u_right = model(x_right)
        bc_loss = (u_left**2 + (u_right - 1.0)**2).mean()

        # Total loss
        loss = pde_loss + bc_loss

        # Backprop
        loss.backward()
        optimizer.step()

        # Print progress every 200 epochs
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}, PDE: {pde_loss.item():.6f}, BC: {bc_loss.item():.6f}")

    # 3f. Test the final result vs. exact solution x^2
    x_test = torch.linspace(0, 1, 11).view(-1, 1)
    with torch.no_grad():
        u_pred = model(x_test).view(-1)
        u_true = x_test.view(-1)**2

    mse_error = torch.mean((u_pred - u_true)**2).item()
    print("Final MSE on 11 test points:", mse_error)

    # Print a small table
    print(" x   |  Prediction  |  Exact(x^2)")
    for x_val, up, ut in zip(x_test, u_pred, u_true):
        print(f"{x_val.item():.2f} | {up.item():.6f}    | {ut.item():.6f}")

if __name__ == "__main__":
    test_stde_1d(STDE)
    pinn_1d_demo()