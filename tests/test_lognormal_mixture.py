"""
Tests for LogNormalMixture: a mixture of lognormal laws, and
LogNormalMixture.from_points(), which builds one from points by kernel smoothing
in their logs.
"""

import pickle
import unittest

import numpy as np
from scipy.integrate import quad

from HARK.distributions import (
    LogNormalMixture,
    LogNormalTruncatedMeanOne,
    expected_continuous,
)


def mixture():
    return LogNormalMixture(
        mu=[-0.3, 0.0, 0.4], sigma=[0.1, 0.25, 0.15], weights=[0.2, 0.5, 0.3], seed=0
    )


def integral(f, law):
    """The integral of f against the law's density, by adaptive quadrature."""
    return quad(lambda x: f(x) * law.pdf(x), 0.0, np.inf, epsabs=1e-14, limit=500)[0]


class testLognormalMixture(unittest.TestCase):
    def test_density_and_moments(self):
        law = mixture()
        self.assertAlmostEqual(integral(lambda x: 1.0, law), 1.0, places=12)
        for n in (1, 2, -2):
            self.assertAlmostEqual(
                law.moment(n) / integral(lambda x: x**n, law), 1.0, places=10
            )
        lo, hi = 0.8, 1.3
        part = quad(lambda x: x * law.pdf(x), lo, hi, epsabs=1e-14)[0]
        self.assertAlmostEqual(float(law.partial_moment(1, lo, hi)), part, places=12)

    def test_cdf_and_quantiles(self):
        law = mixture()
        x = np.linspace(0.3, 3.0, 50)
        exact = [quad(law.pdf, 0.0, xi, epsabs=1e-14)[0] for xi in x]
        np.testing.assert_allclose(law.cdf(x), exact, atol=1e-12)
        q = np.linspace(0.001, 0.999, 101)
        np.testing.assert_allclose(law.cdf(law.ppf(q)), q, atol=1e-13)
        self.assertEqual(law.ppf(0.0), 0.0)
        self.assertEqual(law.ppf(1.0), np.inf)

    def test_equiprobable_discretization(self):
        # the default: N atoms for the whole mixture
        law = mixture()
        for N in (1, 7, 25):
            approx = law.discretize(N, method="equiprobable")
            self.assertAlmostEqual(
                np.dot(approx.pmv, approx.atoms[0]), law.mean(), places=13
            )
            self.assertTrue(np.all(np.diff(approx.atoms[0]) > 0.0))
            self.assertIs(approx.limit["dist"], law)
            self.assertEqual(approx.limit["infimum"][0], 0.0)
            np.testing.assert_array_equal(law.discretize(N).atoms, approx.atoms)

    def test_discretization_by_component(self):
        # N atoms for each component, each with its component's weight over N
        law = mixture()
        for N in (1, 7, 25):
            approx = law.discretize(N, method="components")
            self.assertEqual(approx.atoms[0].size, 3 * N)
            self.assertAlmostEqual(np.sum(approx.pmv), 1.0, places=14)
            self.assertAlmostEqual(
                np.dot(approx.pmv, approx.atoms[0]), law.mean(), places=13
            )
            self.assertTrue(np.all(np.diff(approx.atoms[0]) >= 0.0))
            self.assertEqual(approx.limit["method"], "components")
            self.assertIs(approx.limit["dist"], law)
        for w, part in zip(law.weights, law.components):
            own = part.discretize(25, method="equiprobable").atoms[0]
            carried = approx.pmv[np.isin(approx.atoms[0], own)]
            np.testing.assert_allclose(carried, w / 25, rtol=1e-14)
        # the second moment, understated by the bins' means, converges
        second = law.moment(2)
        gaps = []
        for N in (25, 200):
            approx = law.discretize(N, method="components")
            gaps.append(second - np.dot(approx.pmv, approx.atoms[0] ** 2))
        self.assertTrue(0.0 < gaps[1] < gaps[0])
        self.assertLess(gaps[1] / second, 1e-4)
        ends = law.discretize(4, method="components", endpoints=True)
        self.assertEqual(ends.atoms[0].size, 3 * 4 + 2)
        self.assertEqual(ends.pmv[0] + ends.pmv[-1], 0.0)
        # built from points, a mixture has a component per point
        points = np.linspace(0.6, 1.5, 50)
        smooth = LogNormalMixture.from_points(points)
        self.assertEqual(smooth.discretize(7).atoms[0].size, 7)
        self.assertEqual(smooth.discretize(7, method="components").atoms[0].size, 350)

    def test_draws(self):
        law = mixture()
        draws = law.draw(200000)
        self.assertAlmostEqual(draws.mean() / law.mean(), 1.0, places=2)
        again = LogNormalMixture(
            [-0.3, 0.0, 0.4], [0.1, 0.25, 0.15], [0.2, 0.5, 0.3], seed=0
        )
        np.testing.assert_array_equal(again.draw(10), mixture().draw(10))

    def test_from_points(self):
        rng = np.random.default_rng(3)
        points = LogNormalTruncatedMeanOne.from_sd_log(0.2, -3.0, 3.0).ppf(
            rng.uniform(size=400)
        )
        logs = np.log(points)

        def log_variance(law):
            """Var(ln X): before the truncation, from the components; after it, by quadrature."""
            before = np.var(law.mu) + law.sigma[0] ** 2  # the points' equal weights
            first, second = expected_continuous(
                lambda x: np.stack([np.log(x[0]), np.log(x[0]) ** 2]), law
            )
            return before, second - first**2

        matched = LogNormalMixture.from_points(points)
        self.assertAlmostEqual(matched.mean(), points.mean(), places=13)
        before, after = log_variance(matched)
        self.assertAlmostEqual(before, logs.var(), places=12)
        self.assertAlmostEqual(
            after / logs.var(), 1.0, places=3
        )  # the trim at the ends
        plain = LogNormalMixture.from_points(points, match_variance=False)
        self.assertAlmostEqual(plain.mean(), points.mean(), places=13)
        before, after = log_variance(plain)
        self.assertAlmostEqual(before, logs.var() + plain.sigma[0] ** 2, places=12)
        # the support ends three bandwidths beyond the outermost points, the
        # components' means, exp(mu + h**2 / 2)
        h = matched.sigma[0]
        self.assertAlmostEqual(
            np.log(matched.lower), matched.mu.min() + h * h / 2 - 3 * h, places=12
        )
        self.assertAlmostEqual(
            np.log(matched.upper), matched.mu.max() + h * h / 2 + 3 * h, places=12
        )
        self.assertTrue(np.all(matched.draw(1000) >= matched.lower))
        # Silverman's rule: 0.9 min(s, IQR / 1.34) n ** -0.2
        iqr = np.subtract(*np.percentile(logs, [75, 25]))
        expected_h = 0.9 * min(logs.std(), iqr / 1.34) * 400**-0.2
        self.assertAlmostEqual(matched.sigma[0] / expected_h, 1.0, places=2)

    def test_expected_continuous_component_by_component(self):
        law = mixture()

        def f(x):
            return np.maximum(x[0] - 1.1, 0.0)

        exact = quad(
            lambda x: max(x - 1.1, 0.0) * law.pdf(x), 1.1, np.inf, epsabs=1e-14
        )[0]
        # at four nodes a piece, the default, and at eight
        for nodes, places in ((4, 8), (8, 12)):
            result = expected_continuous(
                f, law, splits=[lambda: np.array([1.1])], nodes=nodes
            )
            self.assertAlmostEqual(float(result), exact, places=places)
            self.assertAlmostEqual(
                expected_continuous(dstns=law, nodes=nodes)[0],
                law.mean(),
                places=places,
            )

    def test_pickle(self):
        law = mixture()
        copy = pickle.loads(pickle.dumps(law))
        x = np.linspace(0.5, 2.0, 7)
        np.testing.assert_array_equal(copy.cdf(x), law.cdf(x))

    def test_errors(self):
        with self.assertRaises(ValueError):
            LogNormalMixture([0.0, 0.1], [0.1, 0.0])
        with self.assertRaises(ValueError):
            LogNormalMixture([0.0, 0.1], [0.1, 0.1], weights=[1.0, -1.0])
        with self.assertRaises(ValueError):
            LogNormalMixture.from_points([1.0, -0.5, 2.0])
        with self.assertRaises(ValueError):
            LogNormalMixture.from_points([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            LogNormalMixture.from_points([0.9, 1.0, 1.1], bandwidth=1.0)
