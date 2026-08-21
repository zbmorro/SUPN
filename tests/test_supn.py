from supn import SUPN
import torch
torch.set_default_dtype(torch.float64)


def test_initialization():
    net = SUPN(max_degree=3, width=10, d=1)
    nparams = sum([param.numel() for param in net.parameters()])
    assert nparams == 50, 'Error in 1D initialization'

    net = SUPN(max_degree=3, width=10, d=2, space='total_degree')
    nparams = sum([param.numel() for param in net.parameters()])
    assert nparams == 110, 'Error in 2D initialization'


def test_differentiation_1d():
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 10001)
    h = x[1] - x[0]
    net = SUPN(max_degree=3, width=10, d=1)
    derivatives = net.dx(x[:, None], order=[1, 2, 3, 4]).reshape(4, x.shape[0])
    y = net(x[:, None]).flatten()

    du_approx = (y[2:] - y[:-2])/(2*h)
    du_err = ((du_approx - derivatives[0, 1:-1]).norm() /
              derivatives[0, 1:-1].norm())
    assert du_err < 4e-6, 'Error in 1D 1st derivative'

    d2_approx = (y[2:] - 2*y[1:-1] + y[:-2])/(h**2)
    d2u_err = ((d2_approx - derivatives[1, 1:-1]).norm() /
               derivatives[1, 1:-1].norm())
    assert d2u_err < 6e-6, 'Error in 1D 2nd derivative'

    d3u_approx = (y[4:] - 2*y[3:-1] + 2*y[1:-3] - y[:-4])/(2 * h**3)
    d3u_err = ((d3u_approx - derivatives[2, 2:-2]).norm() /
               derivatives[2, 2:-2].norm())
    assert d3u_err < 4e-5, 'Error in 1D 3rd derivative'

    d4u_approx = (y[4:] - 4*y[3:-1] + 6*y[2:-2] - 4*y[1:-3] + y[:-4]) / (h**4)
    d4u_err = ((d4u_approx - derivatives[3, 2:-2]).norm() /
               derivatives[3, 2:-2].norm())
    assert d4u_err < 5e-5, 'Error in 1D 4th derivative'


def test_differentiation_2d():
    torch.manual_seed(0)
    x, y = torch.linspace(-1, 3, 1001), torch.linspace(-1, 1, 1001)
    (X, Y) = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    domain_transform = torch.as_tensor([[-1, 3], [-1, 1]])
    net = SUPN(max_degree=3, width=10, d=2, space='total_degree',
               domain_transform=domain_transform)
    derivatives = net.dx(grid, order=[1, 2, 3, 4]).reshape(4, grid.shape[1],
                                                           *X.shape)
    U = net(grid).flatten().reshape(X.shape)

    u_x_approx = (U[2:, :] - U[:-2, :])/(2*dx)
    u_x_err = ((u_x_approx - derivatives[0, 0, 1:-1, :]).norm() /
               derivatives[0, 0, 1:-1, :].norm()).item()
    u_y_approx = (U[:, 2:] - U[:, :-2])/(2*dy)
    u_y_err = ((u_y_approx - derivatives[0, 1, :, 1:-1]).norm() /
               derivatives[0, 0, :, 1:-1].norm()).item()
    assert max(u_x_err, u_y_err) < 8e-5, 'Error in 2D 1st derivative'

    u_xx_approx = (U[2:, :] - 2*U[1:-1, :] + U[:-2, :])/(dx**2)
    u_xx_err = ((u_xx_approx - derivatives[1, 0, 1:-1, :]).norm() /
                derivatives[1, 0, 1:-1, :].norm()).item()
    u_yy_approx = (U[:, 2:] - 2*U[:, 1:-1] + U[:, :-2])/(dy**2)
    u_yy_err = ((u_yy_approx - derivatives[1, 1, :, 1:-1]).norm() /
                derivatives[1, 1, :, 1:-1].norm()).item()
    assert max(u_xx_err, u_yy_err) < 9e-5, 'Error in 2D 2nd derivative'

    u_xxx_approx = ((U[4:, :] - 2*U[3:-1, :] + 2*U[1:-3, :] - U[:-4, :]) /
                    (2 * dx**3))
    u_xxx_err = ((u_xxx_approx - derivatives[2, 0, 2:-2, :]).norm() /
                 derivatives[2, 0, 2:-2, :].norm()).item()
    u_yyy_approx = ((U[:, 4:] - 2*U[:, 3:-1] + 2*U[:, 1:-3] - U[:, :-4]) /
                    (2 * dy**3))
    u_yyy_err = ((u_yyy_approx - derivatives[2, 1, :, 2:-2]).norm() /
                 derivatives[2, 1, :, 2:-2].norm()).item()
    assert max(u_xxx_err, u_yyy_err) < 8e-4, 'Error in 2D 3rd derivative'

    u_xxxx_approx = ((U[4:, :] - 4*U[3:-1, :] + 6*U[2:-2, :] - 4*U[1:-3, :]
                      + U[:-4, :]) / (dx**4))
    u_xxxx_err = ((u_xxxx_approx - derivatives[3, 0, 2:-2, :]).norm() /
                  derivatives[3, 0, 2:-2, :].norm()).item()
    u_yyyy_approx = ((U[:, 4:] - 4*U[:, 3:-1] + 6*U[:, 2:-2] - 4*U[:, 1:-3]
                      + U[:, :-4]) / (dy**4))
    u_yyyy_err = ((u_yyyy_approx - derivatives[3, 1, :, 2:-2]).norm() /
                  derivatives[3, 1, :, 2:-2].norm()).item()
    assert max(u_xxxx_err, u_yyyy_err) < 8e-4, 'Error in 2D 4rd derivative'


def test_precompute_data():
    net = SUPN(max_degree=5, width=1, d=1)
    x = torch.linspace(-1, 1, 101)[:, None]
    net.precompute_data(x)
    cheby_mat = torch.cos(torch.arange(6)[None, :] * torch.arccos(x))
    diff = ((cheby_mat.flatten() -
             net.precomputed_chebyshev_matrix.flatten()).norm())
    assert diff < 1e-13, 'Error in three-term recurrence'

    net = SUPN(max_degree=5, width=1, d=1, x_train=x)
    net.precompute_data(x)
    assert (net(x) == net(x, recompute=True)).all(), 'Error precomputing'

    assert (net._three_term_recurrence(x.flatten(), 0) == 1).all()


def test_domain_transform():
    domain_transform = torch.as_tensor([[2, 7]])
    net = SUPN(max_degree=5, width=1, d=1, domain_transform=domain_transform)
    x = torch.linspace(2, 7, 101)[:, None]
    diff = (net._transform(x).flatten() - torch.linspace(-1, 1, 101)).norm()
    assert diff < 1e-14, 'Error in 1D domain transform'

    domain_transform = torch.as_tensor([[2, 7], [-2, 1]])
    net = SUPN(max_degree=5, width=1, d=2, domain_transform=domain_transform)
    x = torch.linspace(2, 7, 101)
    y = torch.linspace(-2, 1, 51)
    (X, Y) = torch.meshgrid(x, y, indexing='ij')
    x_c = torch.linspace(-1, 1, 101)
    y_c = torch.linspace(-1, 1, 51)
    (X_c, Y_c) = torch.meshgrid(x_c, y_c, indexing='ij')

    grid = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    grid_c = torch.stack((X_c.flatten(), Y_c.flatten()), dim=-1)
    diff = (net._transform(grid).flatten() - grid_c.flatten()).norm()
    assert diff < 1e-13, 'Error in 2D domain transform'

    x = torch.linspace(-1, 1, 51)[:, None]
    net = SUPN(max_degree=3, width=10, d=1)
    derivatives = net.dx(x, order=[1, 2, 3, 4]).reshape(4, x.shape[0])
    net.set_domain_transform(torch.as_tensor([[3, 7]]))
    x = torch.linspace(3, 7, 51)[:, None]
    new_derivatives = net.dx(x, order=[1, 2, 3, 4]).reshape(4, x.shape[0])
    adjustment = 2 ** torch.arange(1, 5)[:, None]
    err = ((new_derivatives * adjustment - derivatives).flatten().norm() /
           derivatives.flatten().norm())
    assert err < 1e-13, 'Error in differentiating affine transform (1D)'

    net.precompute_data(x)
    _derivatives = net.dx(order=[1, 2, 3, 4]).reshape(4, x.shape[0])
    print(new_derivatives, _derivatives)
    assert ((_derivatives-new_derivatives).norm() / new_derivatives.norm()
            < 1e-13), 'Error in precomputation'

    x, y = torch.linspace(-1, 1, 51), torch.linspace(-1, 1, 51)
    (X, Y) = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    net = SUPN(max_degree=3, width=10, d=2, space='total_degree')
    derivatives = net.dx(grid, order=[1, 2, 3, 4]).reshape(4, grid.shape[1],
                                                           *X.shape)
    net.set_domain_transform(torch.as_tensor([[3, 7], [0, 1]]))
    x, y = torch.linspace(3, 7, 51), torch.linspace(0, 1, 51)
    (X, Y) = torch.meshgrid(x, y, indexing='ij')
    grid = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    new_derivatives = net.dx(grid, order=[1, 2, 3, 4])
    new_derivatives = new_derivatives.reshape(4, grid.shape[1], *X.shape)
    adjustment = torch.as_tensor([[2], [0.5]]) ** torch.arange(1, 5)[None, :]
    adjustment = adjustment.T[..., None, None]
    err = ((adjustment * new_derivatives - derivatives).flatten().norm() /
           derivatives.flatten().norm())
    assert err < 1e-13, 'Error in differentiating affine transform (2D)'


def test_exception_handling():
    x = torch.linspace(-1, 1, 101)[:, None]
    net = SUPN(max_degree=5, width=1)

    # Incorrect transform
    try:
        net.set_domain_transform(torch.as_tensor([[1], [2]]))
    except ValueError:
        pass

    # Calling dx without evaluation point or precomputed data
    try:
        net.dx(order=1)
    except RuntimeError:
        pass

    # Derivatives over order 4
    try:
        net.dx(x, order=5)
    except ValueError:
        pass

    # Three-term recurrence with 2D input
    try:
        net._three_term_recurrence(x, 0)
    except ValueError:
        pass

    # Query point outside domain
    try:
        net._transform(2*x)
    except ValueError:
        pass
