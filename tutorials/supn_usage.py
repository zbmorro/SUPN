from supn import SUPN
import torch


# -----------------------
# One-dimensional example
# -----------------------

# Grid must be shape (N_x, d)
x_train = torch.linspace(2, 5, 101)[:, None]
x_val = torch.linspace(2, 5, 98)[:, None]

# Transform must be shape (d, 2)
domain_transform = torch.as_tensor([[2, 5]])

# Initialize and precompute Vandermonde and derivative matrices
net = SUPN(max_degree=5, width=10, domain_transform=domain_transform, d=1)
net.precompute_data(x_train)

# Compute gradients on training grid (grid optional input)
grad_train = net.dx(order=[1, 2, 3, 4])
grad_train = net.dx(x_train, order=[1, 2, 3, 4])

# Compute gradients on validation grid (must supply as input)
grad_val = net.dx(x_val, order=[1, 2, 3, 4])


# -------------------------
# Three-dimensional example
# -------------------------

x = torch.linspace(-1, 1, 101)
y = torch.linspace(-3, 0, 51)
z = torch.linspace(2, 3, 26)
(X, Y, Z) = torch.meshgrid(x, y, z, indexing='ij')

# Shape (N_x, d)
grid = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=-1)

domain_transform = torch.as_tensor([[-1, 1],
                                    [-3, 0],
                                    [2, 3]])

# Options for polynomial degree space: total degree, hyperbolic cross, and
# tensor product
net = SUPN(max_degree=3, width=10, domain_transform=domain_transform, d=3,
           space='total_degree')
net.precompute_data(grid)

grad = net.dx(order=[1, 2])

# grad has shape (len(order), d, N_x, 1) -- last dimension is placeholder for
# num_outputs. We have grad[i, j, k] = d^(i+1) / dx_(j+1)^(i+1) (x_k)

# laplacian = u_xx + u_yy + u_zz
laplacian = grad[1, ...].sum(axis=0)
