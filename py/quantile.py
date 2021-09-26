
import numpy as np


def quantile(k, x):
    """
    Returns the kth largest value in the vector
    """
    assert 1 <= k <= len(x)
    return np.sort(x)[-k]


def most_violated_constraint_simple(k, x, threshold=0.0):
    """
    Returns the coefficients a and b yielding the largest ax + b
    that respects quantile(x') >= ax' + b for all 0 <= x' <= 1.
    If no such a and b exists, return None
    """
    assert 1 <= k <= len(x)
    assert all(0 <= c <= 1 for c in x)
    n = len(x)
    x_s = np.sort(x)[::-1]
    cumulative = np.cumsum(x_s)[k-1:]
    subset_count = np.arange(k, n+1)
    vals = (cumulative - k + 1) / (subset_count - k + 1)
    best = np.argmax(vals)
    coefs = np.ones(n) / (subset_count[best] - k + 1)
    coefs[np.argsort(x)[:-k-best]] = 0
    constant = (1 - k) / (subset_count[best] - k + 1)
    assert np.isclose((coefs * x).sum() + constant, vals[best])
    assert vals[best] <= x_s[k-1] + 1.0e-6
    if vals[best] > threshold:
        return coefs, constant
    return None


def check_constraint(k, x, coefs, constant, tol=1.0e-6):
    bound = np.multiply(coefs, x).sum() + constant
    expected = quantile(k, x)
    assert bound <= expected + tol


def fuzz_constraint(k, coefs, constant, l=0.0, u=1.0, tol=1.0e-6):
    r = l + np.random.rand(len(coefs)) * (u - l)
    assert np.greater_equal(r, l - tol).all()
    assert np.less_equal(r, u + tol).all()
    check_constraint(k, r, coefs, constant, tol)


def _most_violated_constraint_bounded_helper(k, x, u, L, U):
    n = len(x)
    contribs = (x - L) / (np.maximum(u, U) - L)
    c_s = np.sort(contribs)[::-1]
    cumulative = np.cumsum(c_s)[k-1:]
    subset_count = np.arange(k, n+1)
    vals = L + (U - L) * (cumulative - k + 1) / (subset_count - k + 1)
    best = np.argmax(vals)
    coefs = (U - L) * np.ones(n) / (np.maximum(u, U) - L) / (subset_count[best] - k + 1)
    coefs[np.argsort(contribs)[:-k-best]] = 0
    constant = L + (U - L) * (1 - k) / (subset_count[best] - k + 1) - L * coefs.sum()
    assert np.isclose((coefs * x).sum() + constant, vals[best])
    return coefs, constant


def most_violated_constraint(k, x, l, u, threshold=None):
    """
    Returns the coefficients a and b yielding the largest ax + b
    that respects quantile(x') >= ax' + b for all l <= x' <= u.
    If no such a and b exists, return None
    """
    assert 1 <= k <= len(x)
    assert len(l) == len(x)
    assert len(u) == len(x)
    assert np.greater_equal(x, l).all()
    assert np.less_equal(x, u).all()
    n = len(x)
    u_s = np.sort(u)
    L = quantile(k, l)
    best_constraint = None
    best_val = threshold if threshold is not None else L
    for U in np.sort(u_s)[::-1]:
        if U < L:
            continue
        coefs, constant = _most_violated_constraint_bounded_helper(k, x, u, L, U)
        val = np.multiply(coefs, x).sum() + constant
        assert val <= quantile(k, x)
        if val > best_val:
            best_val = val
            best_constraint = (coefs, constant)
    return best_constraint


def test_simple():
    count = 0
    tested = 0
    for n in [2, 3, 4, 6, 10, 20]:
        for k in range(1, n+1):
            for i in range(10):
                constraint = most_violated_constraint_simple(k, np.random.rand(n))
                count += 1
                if constraint is None:
                    continue
                tested += 1
                coefs, constant = constraint
                for j in range(100):
                    fuzz_constraint(k, coefs, constant)
    print(f"{count} attempts, {tested} succesful constraints")


def test_generic():
    count = 0
    tested = 0
    for n in [2, 3, 4, 6, 10, 20]:
        for k in range(1, n+1):
            for i in range(10):
                a = np.random.standard_cauchy(n)
                b = np.random.standard_cauchy(n)
                l = np.minimum(a, b)
                u = np.maximum(a, b)
                x = (u - l) * np.random.rand(n) + l
                constraint = most_violated_constraint(k, x, l, u)
                constraint2 = most_violated_constraint2(k, x, l, u)
                count += 1
                if constraint is None:
                    continue
                assert constraint2 is not None
                tested += 1
                coefs, constant = constraint
                coefs2, constant2 = constraint2
                assert np.isclose(coefs, coefs2).all()
                for j in range(100):
                    fuzz_constraint(k, coefs, constant, l, u)
    print(f"{count} attempts, {tested} succesful constraints")

#test_simple()
#test_generic()
