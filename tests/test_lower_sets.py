from supn import LowerSet
import torch
torch.set_default_dtype(torch.float64)


def test_initialization():
    # Three different spaces
    lower_set = LowerSet(d=1, space='total_degree')
    lower_set.enumerate(5)
    lower_set = LowerSet(d=1, space='hyperbolic_cross')
    lower_set.enumerate(5)
    lower_set = LowerSet(d=1, space='tensor_product')
    lower_set.enumerate(5)

    # Avoid recomputing if everything is the same
    lower_set = LowerSet(d=1, space='tensor_product', max_degree=5)
    lower_set.enumerate(5)
    lower_set.enumerate(5)


def test_enumeration():
    lower_set = LowerSet(d=2, max_degree=3, space='total_degree')
    Lambda = torch.as_tensor([[0, 0], [0, 1], [0, 2], [0, 3],
                              [1, 0], [1, 1], [1, 2],
                              [2, 0], [2, 1],
                              [3, 0]])
    assert (lower_set.enumerate() == Lambda).all(), ('Error enumerating total-'
                                                     'degree space')

    lower_set = LowerSet(d=2, max_degree=3, space='hyperbolic_cross')
    Lambda = torch.as_tensor([[0, 0], [0, 1], [0, 2], [0, 3],
                              [1, 0], [1, 1],
                              [2, 0],
                              [3, 0]])
    assert (lower_set.enumerate() == Lambda).all(), ('Error enumerating hyperb'
                                                     'olic-cross space')

    lower_set = LowerSet(d=2, max_degree=3, space='tensor_product')
    Lambda = torch.as_tensor([[0, 0], [0, 1], [0, 2], [0, 3],
                              [1, 0], [1, 1], [1, 2], [1, 3],
                              [2, 0], [2, 1], [2, 2], [2, 3],
                              [3, 0], [3, 1], [3, 2], [3, 3]])
    assert (lower_set.enumerate() == Lambda).all(), ('Error enumerating tensor'
                                                     '-product space')


def test_augment():
    lower_set = LowerSet(d=2, max_degree=5, space='total_degree')
    lower_set.augment([[4, 3]], inplace=True)
    assert (lower_set.Lambda[-1, :].tolist() == [4, 3])
    lower_set.augment([[2, 6]], A=torch.as_tensor([[1, 2], [2, 3]]))


def test_lower_completion():
    lower_set = LowerSet(d=2, space='total_degree')
    new_A = lower_set.augment([[2, 6]], A=torch.as_tensor([[1, 2], [2, 3]]))
    assert not lower_set.is_lower(new_A)
    lower_completion = lower_set.get_lower_completion(new_A, inplace=True)
    assert lower_set.is_lower(lower_completion)
    assert lower_set.leq(torch.as_tensor([1, 3]), torch.as_tensor([3, 3]))


def test_exception_handling():
    # Invalid space
    try:
        lower_set = LowerSet(d=1, space='total_deg')
    except ValueError:
        pass

    # Enumeration without specified max degree
    try:
        lower_set = LowerSet(d=2, space='total_degree')
        lower_set.enumerate()
    except ValueError:
        pass

    lower_set = LowerSet(d=2, space='total_degree', max_degree=5)

    # Testing admissibility of non-iterable object
    try:
        lower_set.admissible(4)
    except ValueError:
        pass

    # Nothing specified to augment
    try:
        lower_set.augment([4, 2])
    except ValueError:
        pass

    # Augmenting with non-iterable
    try:
        lower_set.augment(4, inplace=True)
    except ValueError:
        pass

    # Incorrect dimensions
    try:
        lower_set.augment([[1, 2, 3]], inplace=True)
    except ValueError:
        pass

    # Calling lower-completion with no target
    try:
        lower_set.get_lower_completion()
    except ValueError:
        pass
