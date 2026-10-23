"""
The IndShock problem with the income shocks integrated against their continuous
laws: the law, the kink-split expectation against brute-force quadrature, and
the solver, whose cubic consumption function stays concave above the kink.
"""

import pickle
import unittest
from copy import deepcopy

import numpy as np
from scipy.integrate import quad

from HARK.Calibration.Income.IncomeProcesses import BufferStockIncShkDstn
from HARK.ConsumptionSaving.ConsIndShockContinuousModel import (
    IndShockContinuousConsumerType,
    KinkSplitExpectation,
    LognormalIncomeLaw,
    _normal_weight,
    _z_bounds,
    convex_segments_above_kink,
    kink_of_solution,
)
from HARK.distributions import truncated_normal_variance
from HARK.interpolation import LinearInterp
from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution


class testLognormalIncomeLaw(unittest.TestCase):
    def test_means_bounds_and_dispersion(self):
        law = LognormalIncomeLaw(0.1, 0.2, 0.05, 0.3)
        kappa = (1 - 0.05 * 0.3) / (1 - 0.05)
        self.assertAlmostEqual(law.kappa, kappa, places=15)
        self.assertAlmostEqual(law.PermShkLaw.moment(1), 1.0, places=13)
        self.assertAlmostEqual(law.TranShkLaw.moment(1), 1.0, places=13)
        self.assertAlmostEqual(law.EmpShkLaw.moment(1), kappa, places=13)
        # The mixture's mean is one: unemployment, and employment at mean kappa
        self.assertAlmostEqual(0.05 * 0.3 + 0.95 * kappa, 1.0, places=15)
        # The stated standard deviations are those of the logs after truncation
        scale = 1 / np.sqrt(truncated_normal_variance(-3.0, 3.0))
        self.assertAlmostEqual(law.PermShkLaw.sigma, 0.1 * scale, places=15)
        self.assertAlmostEqual(law.EmpShkLaw.sigma, 0.2 * scale, places=15)
        self.assertAlmostEqual(scale, 1.013604198, places=9)
        # The bounds, and the inverse permanent shock's
        psi_inf, xi_inf = law.infimum
        self.assertAlmostEqual(psi_inf, np.exp(law.PermShkLaw.mu - 3 * law.PermShkLaw.sigma))
        self.assertEqual(xi_inf, 0.3)
        self.assertAlmostEqual(law.InvPermShkLaw.infimum[0], 1 / law.PermShkLaw.supremum[0])
        self.assertAlmostEqual(law.InvPermShkLaw.moment(1) * 1.0, law.PermShkLaw.moment(-1))
        # The worst income event has no mass unless unemployment pays nothing
        self.assertEqual(law.worst_income_probability(), 0.0)
        self.assertEqual(LognormalIncomeLaw(0.1, 0.1, 0.05, 0.0).worst_income_probability(), 0.05)
        # The natural constraint binds through the infimum of the permanent shock when
        # next period's minimum is below the worst income, and the supremum otherwise
        R, G = 1.03, 1.01
        self.assertAlmostEqual(
            law.natural_borrowing_constraint(0.0, R, G), -0.3 * G * psi_inf / R
        )
        self.assertAlmostEqual(
            law.natural_borrowing_constraint(0.5, R, G),
            0.2 * G * law.PermShkLaw.supremum[0] / R,
        )
        with self.assertRaises(ValueError):
            LognormalIncomeLaw(0.0, 0.1)

    def test_discretization_is_harks_when_untruncated(self):
        law = LognormalIncomeLaw(0.1, 0.1, 0.05, 0.3, (None, None), (None, None))
        dstn = law.discretize(5, 5, seed=0)
        ref = BufferStockIncShkDstn(0.1, 0.1, 5, 5, 0.05, 0.3)
        np.testing.assert_allclose(dstn.atoms, ref.atoms, rtol=0, atol=1e-14)
        np.testing.assert_allclose(dstn.pmv, ref.pmv, rtol=0, atol=1e-15)

    def test_discretization_keeps_the_mean_and_records_the_bounds(self):
        law = LognormalIncomeLaw(0.1, 0.1, 0.05, 0.3)
        for n in (5, 7, 40):
            dstn = law.discretize(n, n)
            self.assertAlmostEqual(
                float(np.sum(dstn.pmv * dstn.atoms[0] * dstn.atoms[1])), 1.0, places=13
            )
            np.testing.assert_allclose(dstn.limit["infimum"], law.infimum, rtol=1e-14)
            self.assertGreater(dstn.atoms[0].min(), law.infimum[0])


def kinked_marginal_values(m_kink, rho):
    """v' = m^(-rho) below the kink and a smooth decreasing function above it,
    continuous at the kink, with its exact derivative."""

    def vp(m):
        m = np.asarray(m, dtype=float)
        x = m - m_kink
        return np.where(
            m <= m_kink, m ** (-rho), m_kink ** (-rho) * np.exp(-0.6 * x) / (1 + 0.1 * x) ** 2
        )

    def vpp(m):
        m = np.asarray(m, dtype=float)
        x = m - m_kink
        above = m_kink ** (-rho) * np.exp(-0.6 * x) * (
            -0.6 / (1 + 0.1 * x) ** 2 - 0.2 / (1 + 0.1 * x) ** 3
        )
        return np.where(m <= m_kink, -rho * m ** (-rho - 1), above)

    return vp, vpp


class testKinkSplitExpectation(unittest.TestCase):
    rho, R, G, m_kink = 2.0, 1.03, 1.01, 0.8694

    def brute_force(self, law, a, f, weight_power):
        """E[psi^(-weight_power) f(R a / (G psi) + theta)] by nested adaptive quadrature,
        split at the kink's preimage in each shock."""
        th, ps = law.EmpShkLaw, law.PermShkLaw
        tz, pz = _z_bounds(th), _z_bounds(ps)
        kw = dict(limit=400, epsabs=1e-14, epsrel=1e-12)

        def over_theta(b):
            if self.m_kink - b > 0:
                z_star = float(np.clip((np.log(self.m_kink - b) - th.mu) / th.sigma, *tz))
            else:
                z_star = tz[0]

            def g(z):
                return f(b + np.exp(th.mu + th.sigma * z)) * _normal_weight(th, z)

            lo = quad(g, tz[0], z_star, **kw)[0] if z_star > tz[0] else 0.0
            hi = quad(g, z_star, tz[1], **kw)[0] if z_star < tz[1] else 0.0
            return lo + hi

        def employed(zp):
            psi = np.exp(ps.mu + ps.sigma * zp)
            return psi ** (-weight_power) * over_theta(self.R * a / (self.G * psi)) * _normal_weight(ps, zp)

        total = quad(employed, pz[0], pz[1], **kw)[0]
        if law.UnempPrb == 0.0:
            return total

        def unemployed(zp):
            psi = np.exp(ps.mu + ps.sigma * zp)
            return psi ** (-weight_power) * f(law.IncUnemp + self.R * a / (self.G * psi)) * _normal_weight(ps, zp)

        points = None
        if a > 0 and self.m_kink > law.IncUnemp:
            z_star = (np.log(self.R * a / (self.G * (self.m_kink - law.IncUnemp))) - ps.mu) / ps.sigma
            if pz[0] < z_star < pz[1]:
                points = [z_star]
        u = quad(unemployed, pz[0], pz[1], points=points, **kw)[0]
        return (1 - law.UnempPrb) * total + law.UnempPrb * u

    def test_matches_brute_force_quadrature(self):
        vp, vpp = kinked_marginal_values(self.m_kink, self.rho)
        a = np.array([0.0, 0.05, 0.27, 1.0, 3.0])
        for law in (LognormalIncomeLaw(0.1, 0.1, 0.05, 0.3), LognormalIncomeLaw(0.05, 0.2, 0.0, 0.0)):
            op = KinkSplitExpectation(law, a, self.R, self.G, self.rho)
            E1, E2 = op.expect(lambda m: (vp(m), vpp(m)), self.m_kink, 0.0)
            B1 = np.array([self.brute_force(law, x, vp, self.rho) for x in a])
            B2 = np.array([self.brute_force(law, x, vpp, self.rho + 1) for x in a])
            np.testing.assert_allclose(E1, B1, rtol=1e-7)
            np.testing.assert_allclose(E2, B2, rtol=3e-8)

    def test_second_order_is_the_derivative_of_the_first(self):
        vp, vpp = kinked_marginal_values(self.m_kink, self.rho)
        law = LognormalIncomeLaw(0.1, 0.1, 0.05, 0.3)
        a, h = np.array([0.05, 0.27, 1.0, 3.0]), 1e-6
        vfun = lambda m: (vp(m), vpp(m))  # noqa: E731
        E1_up = KinkSplitExpectation(law, a + h, self.R, self.G, self.rho).expect(vfun, self.m_kink, 0.0)[0]
        E1_dn = KinkSplitExpectation(law, a - h, self.R, self.G, self.rho).expect(vfun, self.m_kink, 0.0)[0]
        E2 = KinkSplitExpectation(law, a, self.R, self.G, self.rho).expect(vfun, self.m_kink, 0.0)[1]
        np.testing.assert_allclose((E1_up - E1_dn) / (2 * h) * self.G / self.R, E2, rtol=1e-8)

    def test_kink_of_a_solution(self):
        terminal = ConsumerSolution(cFunc=LinearInterp(np.array([0.0, 1.0]), np.array([0.0, 1.0])), mNrmMin=0.0)
        self.assertEqual(kink_of_solution(terminal), np.inf)
        agent = IndShockContinuousConsumerType(cycles=0, verbose=0)
        agent.solve()
        sol = agent.solution[0]
        kink = kink_of_solution(sol)
        unc = sol.cFunc.functions[0]
        # The grid includes the constraint, so a knot sits exactly on the kink
        self.assertAlmostEqual(unc.x_list[1], kink, places=12)
        self.assertAlmostEqual(unc.y_list[1], kink - sol.mNrmMin, places=9)
        self.assertAlmostEqual(float(sol.cFunc(kink)), kink, places=9)


class testIndShockContinuousConsumerType(unittest.TestCase):
    def test_cubic_is_concave_above_the_kink(self):
        for count in (48, 200):
            cubic = IndShockContinuousConsumerType(cycles=0, verbose=0, CubicBool=True, aXtraCount=count)
            cubic.solve()
            segments, convex = convex_segments_above_kink(cubic.solution[0])
            self.assertGreaterEqual(segments, count - 2)
            self.assertEqual(convex, 0)
            self.assertEqual(cubic.solution[0].MPCmax, 1.0)
            self.assertEqual(cubic.solution[0].mNrmMin, 0.0)
        linear = IndShockContinuousConsumerType(cycles=0, verbose=0, aXtraCount=200)
        linear.solve()
        self.assertEqual(convex_segments_above_kink(linear.solution[0])[1], 0)
        m = np.linspace(0.0, 20.0, 2001)
        self.assertLess(np.max(np.abs(cubic.solution[0].cFunc(m) - linear.solution[0].cFunc(m))), 2e-3)
        # Income with a wide transitory shock and no unemployment, on the default grid
        wide = IndShockContinuousConsumerType(
            cycles=0, verbose=0, CubicBool=True, PermShkStd=[0.01], TranShkStd=[0.3], UnempPrb=0.0
        )
        wide.solve()
        self.assertEqual(convex_segments_above_kink(wide.solution[0])[1], 0)

    def test_refusals(self):
        with self.assertRaises(NotImplementedError):
            IndShockContinuousConsumerType(cycles=0, verbose=0, vFuncBool=True).solve()
        with self.assertRaises(ValueError):
            IndShockContinuousConsumerType(cycles=0, verbose=0, BoroCnstArt=None).solve()
        with self.assertRaises(NotImplementedError):
            IndShockContinuousConsumerType(cycles=0, verbose=0, IncUnemp=0.0).solve()

    def test_solution_survives_pickle_and_deepcopy(self):
        agent = IndShockContinuousConsumerType(cycles=0, verbose=0, CubicBool=True)
        agent.solve()
        m = np.linspace(0.0, 20.0, 201)
        c = agent.solution[0].cFunc(m)
        restored = pickle.loads(pickle.dumps(agent.solution[0]))
        np.testing.assert_array_equal(restored.cFunc(m), c)
        np.testing.assert_array_equal(deepcopy(agent).solution[0].cFunc(m), c)
