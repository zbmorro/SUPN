import copy
import torch


class LowerSet:
    def __init__(self, d: int, space: str, max_degree: int = None):
        if space.lower() not in ['total_degree', 'hyperbolic_cross',
                                 'tensor_product']:
            raise ValueError(
                f'Lower sets of type {space} are not implemented.'
                ' Choose one of total_degree, hyperbolic_cross, or tensor'
                ' product.')
        self.d = d
        self.space = space.lower()
        self.max_degree = max_degree
        self.Lambda = None
        if self.max_degree is not None:
            self.Lambda = self.enumerate(self.max_degree)

    def admissible(self, lam):
        if not hasattr(lam, '__iter__'):
            raise ValueError('lam must be iterable')
        if self.space == 'total_degree':
            return sum(lam)+1
        elif self.space == 'hyperbolic_cross':
            out = 1
            for j in lam:
                out *= (j+1)
            return out
        elif self.space == 'tensor_product':
            return max(lam)+1

    def enumerate(self, max_degree: int = None):
        if max_degree is None and self.max_degree is None:
            raise ValueError('Either supply max_degree or set it at '
                             'instantiation')
        # Give priority to supplied argument
        _max_degree = max_degree if max_degree is not None else self.max_degree
        # Don't recompute if everything is the same
        if self.Lambda is not None and _max_degree == self.max_degree:
            return self.Lambda

        index = self.d*[0]
        Lambda = []

        # Keep a running tally where every slot gets kicked up
        # from the last dimension inward. This avoids pruning a large
        # tensor product and keeps Lambda in ascending order too.
        while self.admissible(index) <= _max_degree+1:
            Lambda.append(index)
            index = copy.deepcopy(Lambda[-1])
            pivot = -1
            index[pivot] += 1
            while self.admissible(index) > _max_degree+1:
                index[pivot:] = abs(pivot)*[0]
                pivot -= 1
                if abs(pivot) <= self.d:
                    index[pivot] += 1
            if abs(pivot) > self.d:
                break

        self.Lambda = torch.as_tensor(Lambda)
        return self.Lambda

    def append(self, new_indices, A=None, inplace=False):
        if A is None and (not inplace or self.Lambda is None):
            raise ValueError('Either A must be supplied, or inplace=True')
        if not hasattr(new_indices, '__iter__'):
            raise ValueError('new_indices must be iterable')

        new_A = A.clone() if A is not None else self.Lambda
        assert new_A.ndim == 2 and new_A.shape[1] == self.d
        _new_indices = torch.as_tensor(new_indices)
        _new_indices = (_new_indices[None, :] if _new_indices.ndim == 1 else
                        _new_indices)
        if _new_indices.shape[1] != new_A.shape[1]:
            raise ValueError(
                f'new_indices must be in R^{new_A.shape[1]} but instead'
                f' are in R^{_new_indices.shape[1]}')

        new_A = torch.cat((new_A, _new_indices))

        if inplace:
            self.Lambda = new_A
        return new_A

    def get_lower_completion(self, A=None, inplace=False):
        r'''
        Computes minimal lower set Lambda such that A \subseteq Lambda
        '''
        if A is None and not inplace:
            raise ValueError('Either A must be supplied, or inplace=True')

        new_A = (A.clone() if (inplace is False or self.Lambda is None)
                 else self.Lambda)
        for index in new_A:
            downward_tuple = (torch.meshgrid(*[torch.arange(entry+1) for entry
                              in index], indexing='ij'))
            downward_completion = (torch.stack([tup.flatten() for tup in
                                   downward_tuple], dim=0))

            # Vectorize the search
            matches = (downward_completion.unsqueeze(2) ==
                       new_A.T.unsqueeze(1))
            new_locs = torch.logical_not(matches.all(dim=0).any(dim=-1))

            # Instead of augmenting everything and taking torch.unique at the
            # end, only augment with the new elements.
            if new_locs.any():
                new_A = self.append(downward_completion[..., new_locs].T,
                                    A=new_A)

        if inplace:
            self.Lambda = new_A
        return new_A

    def is_lower(self, A):
        r'''
        Checks whether, for each i \in \Lambda, it holds that

            { j \in \N_0^d : j_k \leq i_k for all k \in [d]} \subset \Lambda

        '''
        return torch.isin(self.get_lower_completion(A), A).all().item()

    def leq(self, x, y):
        r'''
        Checks whether x_k <= y_k for all k \in [d]
        '''
        assert ((x.ndim == 1 or x.shape[1] == 1) and
                (y.ndim == 1 or y.shape[1] == 1))
        return (x.flatten() <= y.flatten()).all().item()
