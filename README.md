# MEGAT-STDE
# Implementation of Stochastic Taylor Derivative Estimator in PyTorch

This repository contains a PyTorch implementation of the **Stochastic Taylor Derivative Estimator (STDE)**, inspired by the winning NeurIPS 2024 paper. This implementation is designed for educational purposes and demonstrates how STDE can be applied to Physics-Informed Neural Networks (PINNs). Note that the PyTorch implementation may not achieve the same computational efficiency as the original JAX-based implementation due to differences in forward-mode automatic differentiation.

---

## **Features**

- **Core STDE Implementation**:
  - A general framework for approximating differential operators using sparse random jets.
- **Example PDE Solvers**:
  - 1D and 2D Poisson equations.
  - Heat equation.
  - Extendable for other PDEs.
- **Educational Focus**:
  - Illustrates challenges and solutions when implementing STDE in PyTorch.

---

## **Getting Started**

### **Prerequisites**

Make sure you have Python 3.8 or later installed along with the following libraries:

- PyTorch
- NumPy

### **Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/maercaestro/megat-stde.git
   cd megat-stde
   ```


2. (Optional) Install the package in editable mode:

   ```bash
   pip install -e .
   ```

---

## **Usage**

### **Run Examples**

1. **Test all Models**:
   This will test STDE with different differential equations, such as Poisson 1D, Poisson 2D and Heat Equation

   ```bash
   python examples/test_models.py
   ```

2. **Compare Timing**:
    This will test STDE on Poisson 1D equation and compare the computational timing for solving with normal pytorch autograd

   ```bash
   python examples/compare_time.py
   ```

3. **cOM**:
    This will test STDE accuracy compared to normal autograd.

   ```bash
   python examples/compare_acc.py
   ```

### **Extend for Custom PDEs**

1. Define the governing equations.
2. Use the `STDE` class to approximate derivatives.
3. Train a neural network model to minimize residuals.

---

## **Project Structure**

```plaintext
megat-stde/
├── src/                    # Core implementation
│   ├── __init__.py
│   ├── stde.py             # STDE class
│   └── models.py           # Multiple equations/models for STDE tests
├── examples/               # Example scripts
│   ├── compare_acc.py      # Compare the accuracy of STDE
│   ├── compare_time.py     # Compare the timing of computation for STDE vs autograd
│   └── test_models.py      # Test STDE agains multiple equations (1D Poisson, 2D etc)
├── futureworks/            # future works for STDE class (maybe implementation using JAX)
│   ├── taylor_ad.py        # taylor mode AD
├── README.md               # Project overview
└── setup.py                # Installation script
```

---

## **Known Issues**

1. **Performance Limitations**:
   - Forward-mode AD in PyTorch is experimental and not as efficient as JAX for this use case.
2. **Not Production-Ready**:
   - This repository is for educational purposes and is not optimized for production environments.
3. **Sparse Derivative Sampling**:
   - Current implementation may have high variance for higher dimensions.

---

## **Future Directions**

1. **Transition to JAX**:
   - Leverage JAX’s efficient forward-mode AD for performance gains.
2. **Extend PDE Examples**:
   - Include wave equations, Burger’s equation, and higher-order PDEs.
3. **Dynamic Optimization**:
   - Explore integration with Liquid Neural Networks (LNNs) for dynamic system optimization.
4. **Taylor-Mode AD**;
   - Explore feasability of Taylor mode AD for pytorch autograd

---

## **Acknowledgments**

This project is based on the principles outlined in the NeurIPS 2024 paper:
**"Stochastic Taylor Derivative Estimator: Efficient amortization for arbitrary differential operators"**.

Special thanks to the PyTorch community and OpenAI for guidance in developing this repository.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contact**

For questions, feedback, or collaboration, please contact:
[Abu Huzaifah Bidin](mailto\:maercaestro@gmail.com)

