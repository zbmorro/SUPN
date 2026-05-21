from typing import Union
import torch
import torch.nn as nn
from .lower_set import LowerSet


class SUPN(nn.Module):
    def __init__(self, max_degree: int, width: int, d: int = 1,
                 space: str = 'total_degree', ntrain: int = 0,
                 domain_transform: torch.Tensor = None) -> None:
        super().__init__()
        lower_set = LowerSet(d=d, space=space)
        self.Lambda = lower_set.enumerate(max_degree)
        self.max_levels_1d = self.Lambda.max(axis=0).values
        self.model = nn.Sequential()
        self.model.append(nn.Linear(self.Lambda.shape[0], width, bias=False))
        self.model.append(nn.Tanh())
        self.model.append(nn.Linear(width, 1, bias=False))
        self.precomputed_chebyshev_matrix = None
        self.T_combination = None
        self.ntrain = ntrain
        self.width = width
        self.d = d
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.set_domain_transform(domain_transform)
        self._reset_parameters()

    def set_domain_transform(self, domain_transform:
                             torch.Tensor = None) -> None:
        r'''
        Map x_k \in [a_k, b_k] to z_k \in [-1, 1] via

            z_k = 2 * (x_k - a_k) / (b_k - a_k) - 1
        '''
        if domain_transform is None:
            self.dilation = torch.ones((self.d,)).to(self.device)
            self.a = -1 * torch.ones((self.d,)).to(self.device)
        elif (
                domain_transform.shape[0] != self.d
                or domain_transform.shape[1] != 2
             ):
            raise ValueError(
                f'Expected torch.Tensor of shape ({self.d}, 2)'
            )
        else:
            self.dilation = (2 / (domain_transform[:, 1] -
                                  domain_transform[:, 0]))
            self.a = domain_transform[:, 0]

    def forward(self, x: torch.Tensor, recompute=False) -> torch.Tensor:
        if (self.precomputed_chebyshev_matrix is not None and x.shape[0] ==
                self.ntrain and not recompute):  # hardcoded NUM_TRAIN_SAMPLES
            chebyshev_mat = self.precomputed_chebyshev_matrix
        else:
            chebyshev_mat = self._form_chebyshev_matrix(self._transform(x))
        return self.model(chebyshev_mat)

    def _compute_chebyshev_data(self, x: torch.Tensor,
                                r_max: int = 4) -> None:
        '''
        Based on differentiating the three-term recurrence for Chebyshev
        polynomials of the first kind

            T_{n+1}^{(r)} = 2x T_n^{(r)} + 2r T_n^{(r-1)} - T_{n-1}^{(r)},
                r>=1, n>=1

        with the base cases

            T_0 = 1, T_1 = x, T_{n+1}^{(0)} = 2x T_n^{(0)} - T_{n-1}^{(0)},
            T_0^{(1)} = 0, T_1^{(1)} = 1

        The final output is T_combination, shape (r_max, d, N_x, M), where

            T_combination[r, k, j, m] = T^{(r)}_{m_k} (x^{(j)}_k)

        i.e. the r-th derivative of degree-m_k Chebyshev polynomial, evaluated
        at component k of node j
        '''

        assert x.ndim == 2 and x.shape[1] == self.d

        # --------------------------------------------------
        # Build cache of basis and derivatives per dimension
        # --------------------------------------------------
        num_recursions = min(r_max+1, self.max_levels_1d.max()+1)
        dpT_cache = torch.zeros((r_max+1, self.d, x.shape[0],
                                 self.max_levels_1d.max()+1)).to(self.device)

        _x = self._transform(x)
        for k in range(self.d):
            n = self.max_levels_1d[k] + 1

            # T_n
            dpT_cache[0, k, :, :n] = self._three_term_recurrence(_x[:, k], n-1)

            if num_recursions > 1:
                # d^r (T_n)/ dx^r
                for r in range(1, num_recursions):
                    for j in range(r, n):
                        if r == 1 and j == 1:
                            # T_1 is specified separately in recurrence
                            dpT_cache[r, k, :, j] = 1.0
                        else:
                            dpT_cache[r, k, :, j] = (
                                dpT_cache[r, k, :, j-1] * 2 * _x[:, k] +
                                2 * r * dpT_cache[r-1, k, :, j-1] -
                                dpT_cache[r, k, :, j-2])

        # -------------------------------------------------------------
        # Now, package everything ordered according to Lambda index set
        # shape (r_max, n_dim, n_discretization, M)
        # -------------------------------------------------------------
        T = torch.zeros((dpT_cache.shape[0], self.d, x.shape[0],
                         self.Lambda.shape[0])).to(self.device)
        for i in range(self.Lambda.shape[0]):
            for k in range(self.d):
                n = self.Lambda[i, k]
                T[:, k, :, i] = dpT_cache[:, k, :, n]

        # ----------------------------------------------------------
        # Differentiate each component in product of basis functions
        # ----------------------------------------------------------
        T_combination = T.clone()
        for k in range(self.d):
            for j in [*range(k), *range(k+1, self.d)]:
                for r in range(T.shape[0]):
                    T_combination[r, k, ...] = (
                        T_combination[r, k, ...] * T[0, j, ...])

        # ----------------------------------------
        # Adjust T^{(r)} for affine transformation
        # ----------------------------------------
        adjustment = torch.stack([self.dilation**k for k in
                                  range(T.shape[0])], axis=0)
        return T_combination * adjustment[..., None, None]

    def precompute_data(self, x: torch.Tensor, r_max: int = 4) -> None:
        assert x.ndim == 2 and x.shape[1] == self.d
        self.ntrain = x.shape[0]
        self.precomputed_chebyshev_matrix = (
            self._form_chebyshev_matrix(self._transform(x)))
        self.T_combination = self._compute_chebyshev_data(x, r_max)

    def dx(self, x: torch.Tensor = None, order: Union[int, list[int]] = 1):
        r'''
        Returns array of derivatives, size (d, n_x, n_out)

        Based on f(x) = w^T \tanh (A T(x)) and d(tanh(z))/dz = 1-tanh^2(z)
        '''
        if self.precomputed_chebyshev_matrix is None and x is None:
            raise RuntimeError(
                'Must supply `x`, or call supn.precompute_data '
                'before supn.dx')

        A = []
        S = []
        res = []
        _order = order if hasattr(order, '__iter__') else [order]
        max_order = max(_order)
        if max_order > 4:
            raise ValueError(
                'Only analytical derivatives up to order 4 are implemented')

        T_combination = (self.T_combination if
                         ((x is None or x.shape[0] == self.ntrain) and
                          self.T_combination is not None)
                         else self._compute_chebyshev_data(x, max_order))

        # --------------------------------------------------------------------
        # A[r, k, i, :] = \mat{A} T_combination[r, k, i, :]
        # S[r, k, i, j] = d^r / dx^r (\sigma( \prod_{k=1}^d T_{j_k} )|_{
        #     x=x^{(i)}_k}
        # --------------------------------------------------------------------
        A.append(self.model[0](T_combination[0]))
        S.append(self.model[1](A[0]))

        # ------------------------------------------
        # \sigma'(x) = d/dx( tanh(x) ) = 1-tanh(x)^2
        # ------------------------------------------
        A.append(self.model[0](T_combination[1]))
        S.append(1-S[0]**2)

        # ---------------------------------
        # Only compute necessary quantities
        # ---------------------------------
        if 1 in _order:
            res.append(self.model[2](S[1]*A[1]).to(self.device))
        if 2 <= max_order:
            A.append(self.model[0](T_combination[2]))
            S.append(-2*S[0]*S[1])
            if 2 in _order:
                res.append(self.model[2](S[1]*A[2] + S[2]*A[1]**2).to(
                    self.device))
        if 3 <= max_order:
            A.append(self.model[0](T_combination[3]))
            S.append(-2*(S[0]*S[2] + S[1]**2))
            if 3 in _order:
                res.append(self.model[2](S[1]*A[3] + 3*S[2]*A[1]*A[2]
                           + S[3]*A[1]**3).to(self.device))
        if 4 in _order:
            A.append(self.model[0](T_combination[4]))
            S.append(-2*(S[0]*S[3] + 3*S[1]*S[2]))
            res.append(self.model[2](S[1]*A[4] + 4*S[2]*A[1]*A[3]
                       + 3*S[2]*A[2]**2 + 6*S[3]*A[2]*A[1]**2 + S[4]*A[1]**4
                      ).to(self.device))
        return torch.stack(res, dim=0) if len(res) > 1 else res[0]

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Transform [a_1, b_1] \times \dots \times [a_d, b_d] to [-1, 1]^d
        x: shape (N, d)
        '''
        return (x - self.a) * self.dilation - 1.0

    def _reset_parameters(self) -> None:
        r'''
        Reset network parameters from a Kaiming uniform distribution
        '''
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def _three_term_recurrence(self, x: torch.Tensor, max_deg: int):
        r'''
        T_0 = 1
        T_1 = x
        T_{n+1} = 2x T_n - T_{n-1},    n >= 1
        '''
        if x.ndim != 1:
            raise ValueError('x must be 1D torch.Tensor')
        cache = torch.ones((x.shape[0], 1)).to(self.device)
        if max_deg == 0:
            return cache
        cache = torch.cat((cache, x[:, None]), axis=1)
        for j in range(1, max_deg):
            cache = torch.cat(
                (cache, (2*x*cache[:, -1] - cache[:, -2])[:, None]), axis=1)
        return cache

    def _form_chebyshev_matrix(self, x):
        assert x.ndim == 2 and x.shape[1] == self.d
        chebyshev_mat = torch.ones((x.shape[0], self.Lambda.shape[0]))
        chebyshev_cache = torch.zeros((self.d, x.shape[0],
                                       self.max_levels_1d.max()+1))

        # Compute three-term recurrence outside main for-loop
        for k in range(self.d):
            chebyshev_cache[k, :, :self.max_levels_1d[k]+1] = (
                self._three_term_recurrence(x[:, k], self.max_levels_1d[k]))

        # Product of 1D basis functions from indices in Lambda
        for i in range(self.Lambda.shape[0]):
            for k in range(self.d):
                n = self.Lambda[i, k]
                chebyshev_mat[:, i] *= chebyshev_cache[k, :, n]

        return chebyshev_mat.to(self.device)
