"""
Tests for truncated income shocks in IndShock's income constructor: the bounds
PermShk_inf, PermShk_sup, TranShk_inf and TranShk_sup, in standard deviations of
the underlying normal, with None (the default) for no bound.
"""

import unittest

import numpy as np

from HARK.Calibration.Income.IncomeProcesses import (
    BufferStockIncShkDstn,
    LognormPermIncShk,
    MixtureTranIncShk,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)
from HARK.distributions import LogNormalTruncatedMeanOne, truncated_normal_variance

BOUNDS = {
    "PermShk_inf": -3.0,
    "PermShk_sup": 3.0,
    "TranShk_inf": -3.0,
    "TranShk_sup": 3.0,
}


class testTruncatedIncomeShocks(unittest.TestCase):
    def test_no_bounds_is_the_untruncated_process(self):
        plain = BufferStockIncShkDstn(0.1, 0.1, 7, 7, 0.05, 0.3)
        none = BufferStockIncShkDstn(
            0.1, 0.1, 7, 7, 0.05, 0.3, **{key: None for key in BOUNDS}
        )
        np.testing.assert_array_equal(plain.atoms, none.atoms)
        np.testing.assert_array_equal(plain.pmv, none.pmv)
        agent = IndShockConsumerType()
        self.assertEqual(agent.IncShkDstn.conditional.keys() & BOUNDS.keys(), set())

    def test_permanent_shock(self):
        shock = LognormPermIncShk(0.1, 7, bot=-3.0, top=3.0)
        law = shock.limit["dist"]
        self.assertIsInstance(law, LogNormalTruncatedMeanOne)
        # the log's standard deviation after the truncation is the stated one
        self.assertAlmostEqual(
            law.sigma**2 * truncated_normal_variance(-3.0, 3.0), 0.01, places=15
        )
        expected = law.discretize(7, method="equiprobable")
        np.testing.assert_array_equal(shock.atoms, expected.atoms)
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=15)
        # the recorded bounds are the law's: 0.734122 and 1.348620 at sigma = 0.1
        self.assertAlmostEqual(shock.limit["infimum"][0], 0.734122, places=6)
        self.assertAlmostEqual(shock.limit["supremum"][0], 1.348620, places=6)
        self.assertTrue(
            np.all(shock.atoms > shock.limit["infimum"])
            and np.all(shock.atoms < shock.limit["supremum"])
        )

    def test_one_sided_bound(self):
        shock = LognormPermIncShk(0.1, 7, bot=-2.0)
        self.assertEqual(shock.limit["supremum"][0], np.inf)
        self.assertGreater(shock.limit["infimum"][0], 0.0)
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=15)

    def test_transitory_shock_with_unemployment(self):
        shock = MixtureTranIncShk(0.1, 0.05, 0.3, 7, bot=-3.0, top=3.0)
        employed = LogNormalTruncatedMeanOne.from_sd_log(0.1, -3.0, 3.0).discretize(
            7, method="equiprobable"
        )
        scale = (1.0 - 0.05 * 0.3) / (1.0 - 0.05)
        np.testing.assert_allclose(
            shock.atoms[0], np.r_[0.3, scale * employed.atoms[0]], rtol=1e-15
        )
        np.testing.assert_allclose(shock.pmv, np.r_[0.05, np.full(7, 0.95 / 7)])
        self.assertAlmostEqual(np.dot(shock.pmv, shock.atoms[0]), 1.0, places=15)
        self.assertEqual(shock.limit["infimum"][0], 0.3)

    def test_agent(self):
        agent = IndShockConsumerType(cycles=0, **BOUNDS)
        shocks = agent.IncShkDstn[0]
        perm, tran = shocks.atoms
        self.assertAlmostEqual(np.dot(shocks.pmv, perm), 1.0, places=14)
        self.assertAlmostEqual(np.dot(shocks.pmv, tran), 1.0, places=14)
        self.assertTrue(np.all(perm > 0.734122) and np.all(perm < 1.348620))
        agent.solve()
        self.assertTrue(np.isfinite(agent.solution[0].cFunc(1.0)))

    def test_lifecycle(self):
        # the bounds apply in every working period; retirement keeps its points
        agent = IndShockConsumerType(**{**init_lifecycle, **BOUNDS})
        plain = IndShockConsumerType(**init_lifecycle)
        for t in range(agent.T_cycle):
            sigma = init_lifecycle["PermShkStd"][t]
            perm = agent.IncShkDstn[t].atoms[0]
            if sigma > 0.0:
                law = LogNormalTruncatedMeanOne.from_sd_log(sigma, -3.0, 3.0)
                self.assertTrue(np.all(perm >= law.infimum[0]))
            else:
                np.testing.assert_array_equal(perm, plain.IncShkDstn[t].atoms[0])
