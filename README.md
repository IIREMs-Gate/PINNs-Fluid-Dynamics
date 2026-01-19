# Physics-Informed Neural Network (PINN) for Navier-Stokes


**Project Objective:** Modeling fluid velocity fields using PINNs constrained by Navier-Stokes equations.

## Setup 
Run `pip install -r requirements.txt` to install dependencies. 

Run `python pinn.py` to start training.


## Model Architecture (PyTorch Implementation)
The model is implemented as a deep Feed-Forward Neural Network (MLP) using the torch.nn module.
* **Why this architecture?** MLPs are effective for PINNs because they provide a continuous, differentiable mapping from spatio-temporal coordinates $(x, z, t)$ to fluid variables $(u, w, p)$. This is essential for calculating exact derivatives via automatic differentiation.

### 1. Network Topology
Input Layer: 3 neurons representing $(x, z, t)$ coordinates.

Hidden Layers: 3 fully connected layers with 128 neurons each.

Output Layer: 3 neurons representing the fluid state: $(u, w, p)$.
* $u$: Velocity in the x-direction.
* $w$: Velocity in the z-direction.
* $p$: Pressure field.

### 2. Activation Function: SiLU (Swish)
We utilize nn.SiLU() (Sigmoid Linear Unit) as the activation function. While Tanh is classic, SiLU often allows for smoother gradient flow and can help the model converge faster in complex fluid simulations.

### 3. Physics Integration
The architecture is "Physics-Informed" because the output $u, w, p$ is passed into a custom loss function that computes:
$$Loss_{total} = Loss_{Data} + Loss_{Physics}$$

1.  **Data Loss (MSE):** Measures the error between predicted velocity and the ground truth from the `.tsv` datasets at specific time steps.
2.  **Physics Loss (Residual):** The mean squared residual of the **Incompressible Navier-Stokes equations**:
    * Continuity: $u_x + w_z = 0$
    * Momentum: $u_t + (u \cdot \nabla)u = -\frac{1}{\rho}\nabla p + \nu \nabla^2 u$


## 3. Data Pipeline
The data is ingested using a custom PyTorch `Dataset` and `DataLoader` (see `pinn.py`). 
* **Pre-processing:** Points are extracted from `.tsv` files, converted to Tensors, and moved to the GPU (if available).
* **Sampling:** The pipeline combines labeled data points for supervised learning and unlabeled coordinate points for physics-informed training.

## 4. Results
The model generates a comparison between the predicted velocity magnitude and the JHTDB ground truth.
<img width="1122" height="451" alt="Results" src="https://github.com/user-attachments/assets/b86b7a2c-303f-4b30-bafa-c6c66617f469" />
