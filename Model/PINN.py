import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NU_FIXED = 0.000185     # Constant viscosity provided
print(f"Device: {DEVICE} | Nu: {NU_FIXED}")

# --- BLOCK 1: DATA LOADING ---
df0 = pd.read_csv('./Data/isotropic1024coarse_t0.tsv', sep='\t', skiprows=1) # t0
df1 = pd.read_csv('./Data/isotropic1024coarse_t0_02.tsv', sep='\t', skiprows=1) # t1

t0_val = df0['time'].iloc[0]
t1_val = df1['time'].iloc[0]

class FetchDataset(Dataset):
    def __init__(self, df, t_val):
        self.x = torch.tensor(df['x_point'].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
        self.z = torch.tensor(df['z_point'].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
        self.t = torch.ones_like(self.x) * t_val
        self.u = torch.tensor(df['ux'].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
        self.w = torch.tensor(df['uz'].values, dtype=torch.float32).view(-1, 1).to(DEVICE)

    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self, idx): 
        return self.x[idx], self.z[idx], self.t[idx], self.u[idx], self.w[idx]

# DataLoader for Initial Condition (t=0)
ic_loader = DataLoader(FetchDataset(df0, t0_val), batch_size=2048, shuffle=True)

# for sampling collocation points across the XZ-T domain
def get_collocation_xz(n, x_range, z_range, t_range):
    x = (torch.rand(n, 1).to(DEVICE) * (x_range[1] - x_range[0])) + x_range[0]
    z = (torch.rand(n, 1).to(DEVICE) * (z_range[1] - z_range[0])) + z_range[0]
    t = (torch.rand(n, 1).to(DEVICE) * (t_range[1] - t_range[0])) + t_range[0]
    return x, z, t

x_range = [df0['x_point'].min(), df0['x_point'].max()]
z_range = [df0['z_point'].min(), df0['z_point'].max()]
t_range = [t0_val, t1_val]

# --- BLOCK 2: SIMPLIFIED MODEL (No Fourier, No Scaler) ---
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Direct input (x, z, t) -> 3 features
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 3) # Outputs: u, w, p
        )

    def forward(self, x, z, t):
        return self.net(torch.cat([x, z, t], dim=1))

# --- BLOCK 3: PHYSICS ENGINE (XZ Plane) ---
def get_physics_loss(model, x, z, t, nu, rho=1.0):
    x.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)

    out = model(x, z, t)
    u, w, p = out[:,0:1], out[:,1:2], out[:,2:3]

    # First Order Derivatives
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]
    w_x = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, torch.ones_like(w), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, torch.ones_like(p), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    w_t = torch.autograd.grad(w, t, torch.ones_like(w), create_graph=True)[0]

    # Seconds Order Derivatives
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, torch.ones_like(u_z), create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, torch.ones_like(w_x), create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, torch.ones_like(w_z), create_graph=True)[0]

    # Navier-Stokes residuals
    # Momentum in x and z
    res_mx = u_t + (u*u_x + w*u_z) + (1/rho)*p_x - nu*(u_xx + u_zz)
    res_mz = w_t + (u*w_x + w*w_z) + (1/rho)*p_z - nu*(w_xx + w_zz)


    res_con = u_x + w_z

    return torch.mean(res_con**2) + torch.mean(res_mx**2) + torch.mean(res_mz**2)



# --- BLOCK 4: TRAINING ---
model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n--- Training Starting ---")
for epoch in range(5001):
    optimizer.zero_grad()

    # 1. IC Loss (Data matching at t0)
    batch = next(iter(ic_loader))
    x_i, z_i, t_i, u_i, w_i = batch
    out_ic = model(x_i, z_i, t_i)
    loss_ic = torch.mean((out_ic[:, 0:1] - u_i)**2 + (out_ic[:, 1:2] - w_i)**2)

    # 2. Physics Loss (Enforcing laws from t0 to t1)
    x_c, z_c, t_c = get_collocation_xz(4096, x_range, z_range, t_range)
    loss_phys = get_physics_loss(model, x_c, z_c, t_c, NU_FIXED)

    total_loss = loss_ic + 1.0 * loss_phys
    total_loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch: {epoch} | Total Loss: {total_loss.item():.6f} | IC: {loss_ic.item():.6f} | Phys: {loss_phys.item():.6f}")



# --- BLOCK 5: VISUALIZATION ---
def plot_results(model, df_truth, t_val):
    model.eval()
    n = 100
    x = np.linspace(x_range[0], x_range[1], n)
    z = np.linspace(z_range[0], z_range[1], n)
    X, Z = np.meshgrid(x, z)

    x_in = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(DEVICE)
    z_in = torch.tensor(Z.flatten(), dtype=torch.float32).view(-1, 1).to(DEVICE)
    t_in = torch.ones_like(x_in) * t_val

    with torch.no_grad():
        out = model(x_in, z_in, t_in)
        mag_pred = torch.sqrt(out[:,0]**2 + out[:,1]**2).cpu().numpy().reshape(n, n)

    # Calculate magnitude directly from df_truth
    mag_truth = np.sqrt(df_truth['ux']**2 + df_truth['uz']**2).values
    triang = tri.Triangulation(df_truth['x_point'].values, df_truth['z_point'].values)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    # Use tricontourf for unstructured data
    im1 = ax[0].tricontourf(triang, mag_truth, levels=50, cmap='inferno')
    ax[0].set_title(f"Truth Magnitude (t={t_val})")
    plt.colorbar(im1, ax=ax[0])

    im2 = ax[1].contourf(X, Z, mag_pred, levels=50, cmap='inferno')
    ax[1].set_title(f"PINN Magnitude (t={t_val})")
    plt.colorbar(im2, ax=ax[1])
    plt.show()

plot_results(model, df1, t1_val)