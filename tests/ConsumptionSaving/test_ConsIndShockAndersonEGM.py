"""Tests for the opt-in Anderson-accelerated EGM solver.

Validates that ``IndShockConsumerTypeAndersonEGM`` reaches the SAME stationary
fixed point as plain EGM (run to true convergence), in materially fewer sweeps,
and that it never perturbs the default (non-opt-in) solution path.
"""

import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
    solve_one_period_ConsIndShock,
)
from HARK.ConsumptionSaving.ConsIndShockModelAndersonEGM import (
    IndShockConsumerTypeAndersonEGM,
    anderson_accelerate,
)


def brute_picard_to_convergence(d, n=12000, tol=1e-13):
    """Plain EGM iterated to TRUE convergence.  (The default infinite-horizon
    solve stops on its cFunc-distance criterion before the slowly-converging
    hNrm/MPCmin tail settles, so for an exact parity reference we iterate the
    stock per-period solver until the policy stops moving.)"""
    agent = IndShockConsumerType(**d)
    agent.pre_solve()
    args = (agent.IncShkDstn[0], agent.LivPrb[0], agent.DiscFac, agent.CRRA,
            agent.Rfree[0], agent.PermGroFac[0], agent.BoroCnstArt,
            agent.aXtraGrid, agent.vFuncBool, agent.CubicBool)
    sol = agent.solution_terminal
    m_chk = sol.mNrmMin if hasattr(sol, "mNrmMin") else 0.0
    m_chk = np.linspace(0.5, 60.0, 200)
    prev = None
    for _ in range(n):
        sol = solve_one_period_ConsIndShock(sol, *args)
        c = sol.cFunc(m_chk)
        if prev is not None and np.max(np.abs(c - prev)) < tol:
            break
        prev = c
    return sol


def base_dict(**over):
    d = dict(init_idiosyncratic_shocks)
    d["cycles"] = 0
    d["DiscFac"] = 0.96
    d.update(over)
    return d


class TestAndersonEGMParity(unittest.TestCase):
    def _check(self, d, places_c=8):
        ref = brute_picard_to_convergence(d)
        agent = IndShockConsumerTypeAndersonEGM(**d)
        agent.solve()
        sol = agent.solution[0]
        m = ref.mNrmMin + np.linspace(0.02, 45.0, 400)
        self.assertLess(float(np.max(np.abs(sol.cFunc(m) - ref.cFunc(m)))),
                        10.0 ** (-places_c))
        self.assertLess(float(np.max(np.abs(sol.vPfunc(m) - ref.vPfunc(m)))),
                        1e-7)
        self.assertAlmostEqual(sol.hNrm, ref.hNrm, places=6)
        self.assertAlmostEqual(sol.MPCmin, ref.MPCmin, places=8)
        self.assertAlmostEqual(sol.MPCmax, ref.MPCmax, places=6)
        self.assertAlmostEqual(sol.mNrmMin, ref.mNrmMin, places=8)
        self.assertTrue(agent.last_anderson_info["converged"])

    def test_linear_parity(self):
        self._check(base_dict(CubicBool=False, vFuncBool=False))

    def test_cubic_parity(self):
        self._check(base_dict(CubicBool=True, vFuncBool=False))

    def test_vfunc_parity(self):
        d = base_dict(CubicBool=False, vFuncBool=True)
        ref = brute_picard_to_convergence(d)
        agent = IndShockConsumerTypeAndersonEGM(**d)
        agent.solve()
        sol = agent.solution[0]
        m = ref.mNrmMin + np.linspace(0.05, 40.0, 300)
        self.assertLess(float(np.max(np.abs(sol.vFunc(m) - ref.vFunc(m)))), 1e-6)

    def test_iteration_reduction_at_edge(self):
        """At the patience edge Anderson must take materially fewer sweeps than
        plain Picard (within this solver) to reach the same fixed point."""
        d = base_dict(DiscFac=0.99, aXtraMax=150, aXtraCount=64, quiet=True)
        picard = IndShockConsumerTypeAndersonEGM(anderson_accelerate=False, **d)
        picard.solve()
        ander = IndShockConsumerTypeAndersonEGM(anderson_accelerate=True, **d)
        ander.solve()
        m = picard.solution[0].mNrmMin + np.linspace(0.02, 120, 400)
        self.assertLess(
            float(np.max(np.abs(ander.solution[0].cFunc(m)
                                - picard.solution[0].cFunc(m)))), 1e-6)
        self.assertLess(ander.last_anderson_info["iters"],
                        0.5 * picard.last_anderson_info["iters"])


class TestAndersonEGMGuards(unittest.TestCase):
    def test_cycles_not_zero_raises(self):
        agent = IndShockConsumerTypeAndersonEGM(**base_dict(cycles=1))
        with self.assertRaises(NotImplementedError):
            agent.solve()

    def test_default_path_unchanged(self):
        """Constructing/solving the accelerated type must not alter the stock
        type's default solution."""
        d = base_dict()
        stock_a = IndShockConsumerType(**d)
        stock_a.solve()
        _ = IndShockConsumerTypeAndersonEGM(**d).solve()
        stock_b = IndShockConsumerType(**d)
        stock_b.solve()
        m = np.linspace(0.1, 30.0, 200)
        self.assertTrue(np.allclose(stock_a.solution[0].cFunc(m),
                                    stock_b.solution[0].cFunc(m)))


class TestAndersonDriver(unittest.TestCase):
    def test_accelerates_linear_contraction(self):
        """Anderson on a simple linear contraction matches the fixed point and
        beats plain Picard's iteration count."""
        A = 0.995  # near-unit contraction (slow Picard)
        b = np.linspace(1.0, 2.0, 20)
        x_fp = b / (1.0 - A)

        def sweep(x):
            return A * x + b

        x_and, info = anderson_accelerate(sweep, np.zeros_like(b),
                                          depth=5, tol=1e-12, maxit=10000)
        self.assertTrue(info["converged"])
        self.assertTrue(np.allclose(x_and, x_fp, atol=1e-6))
        self.assertLess(info["iters"], 100)  # Picard would need ~5000


if __name__ == "__main__":
    unittest.main()
