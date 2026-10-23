"""
The pairing rule: cubic Hermite interpolation of the consumption function is
refused when the expectation over the income shocks is a sum over discrete
atoms, which every HARK solver takes. The demonstration solves with the check
disabled, which no user can do, and measures what the rule prevents.
"""

import unittest
from unittest import mock

import numpy as np
from scipy.optimize import brentq

from HARK.ConsumptionSaving.ConsIndShockModel import (
    CUBIC_PAIRING_MESSAGE,
    IndShockConsumerType,
    KinkedRconsumerType,
    check_cubic_pairing,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType


def concavity_violations_above_the_kink(agent):
    """
    Solve the agent and inspect its unconstrained consumption function above this
    period's kink, where the lower envelope uses the cubic rather than the
    constrained line. Returns the number of segments wholly or partly above the
    kink, the number of those where the cubic violates concavity (its second
    derivative is positive at either end, computed exactly from the knots' values
    and slopes), how many of those segments hold an image of next period's kink (an
    asset level from which some atom lands exactly on it), and for linear
    interpolation the number of knots above the kink where the chord slopes rise.
    """
    agent.solve()
    sol = agent.solution[0]
    unc = sol.cFunc.functions[0]
    m, c = np.asarray(unc.x_list), np.asarray(unc.y_list)
    kink = brentq(lambda x: unc(x) - (x - sol.mNrmMin), m[1], m[-1])
    above = m[1:] > kink
    h, slope = np.diff(m), np.diff(c) / np.diff(m)
    if not agent.CubicBool:
        rises = np.diff(slope)[above[1:]] > 1e-12
        return int(above.sum()), 0, 0, int(rises.sum())
    s = np.asarray(unc.dydx_list)
    c2 = np.maximum(
        (6 * slope - 4 * s[:-1] - 2 * s[1:]) / h,
        (-6 * slope + 2 * s[:-1] + 4 * s[1:]) / h,
    )
    violating = np.flatnonzero((c2 > 1e-9) & above)
    psi, theta = agent.IncShkDstn[0].atoms
    images = (kink - theta) * agent.PermGroFac[0] * psi / agent.Rfree[0]
    a = m - c
    held = sum(((images > a[k]) & (images < a[k + 1])).any() for k in violating)
    return int(above.sum()), int(violating.size), int(held), 0


class testCubicPairingRule(unittest.TestCase):
    def test_refused_with_the_explanation(self):
        for make in (
            lambda: IndShockConsumerType(CubicBool=True),
            lambda: KinkedRconsumerType(CubicBool=True),
            lambda: MarkovConsumerType(CubicBool=True),
        ):
            agent = make()
            with self.assertRaises(ValueError) as caught:
                agent.solve()
            self.assertEqual(str(caught.exception), CUBIC_PAIRING_MESSAGE)
        # The helper itself, and the pairing that is allowed
        with self.assertRaises(ValueError):
            check_cubic_pairing(True)
        check_cubic_pairing(True, continuous=True)
        check_cubic_pairing(False)

    def test_linear_is_unaffected(self):
        agent = IndShockConsumerType(cycles=0)
        agent.solve()
        self.assertFalse(agent.CubicBool)
        self.assertAlmostEqual(agent.solution[0].cFunc(1.0), 0.86524, places=4)

    @mock.patch(
        "HARK.ConsumptionSaving.ConsIndShockModel.check_cubic_pairing",
        lambda *args, **kwargs: None,
    )
    def test_what_the_rule_prevents(self):
        # HARK's default model with the artificial constraint at zero, on a grid
        # fine enough to isolate the images of next period's kink: the cubic
        # violates concavity in segments holding an image, and the linear chords
        # never rise
        above, violating, held, _ = concavity_violations_above_the_kink(
            IndShockConsumerType(cycles=0, CubicBool=True, aXtraCount=200)
        )
        self.assertGreater(above, 100)
        self.assertGreaterEqual(violating, 1)
        self.assertEqual(held, violating)
        above, _, _, rises = concavity_violations_above_the_kink(
            IndShockConsumerType(cycles=0, CubicBool=False, aXtraCount=200)
        )
        self.assertGreater(above, 100)
        self.assertEqual(rises, 0)
        # The same with i.i.d. income on the default grid
        kwds = dict(PermShkStd=[0.0], PermShkCount=1, UnempPrb=0.0, TranShkStd=[0.3])
        above, violating, held, _ = concavity_violations_above_the_kink(
            IndShockConsumerType(cycles=0, CubicBool=True, **kwds)
        )
        self.assertGreaterEqual(violating, 1)
        self.assertEqual(held, violating)
        _, _, _, rises = concavity_violations_above_the_kink(
            IndShockConsumerType(cycles=0, CubicBool=False, **kwds)
        )
        self.assertEqual(rises, 0)
