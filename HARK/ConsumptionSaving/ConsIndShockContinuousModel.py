"""
The IndShock problem with the expectation over the income shocks taken against
their continuous laws rather than summed over discrete atoms.

HARK's :func:`solve_one_period_ConsIndShock` takes next period's expected marginal
value as a sum over the atoms of ``IncShkDstn``. Each atom sends next period's
borrowing-constraint kink to one point of this period's asset grid, where this
period's consumption function then has a kink of its own. A cubic Hermite segment
is smooth and cannot follow a kink inside it, so wherever a segment holds one of
these images the consumption function violates concavity; the finer the grid, the
more concavity violations show. Integrating against the laws' densities removes
the images: the expectation is then a smooth function of assets, and cubic Hermite
interpolation with the endogenous grid method's exact slopes represents it
faithfully.

The method here follows the one built for HAFiscal. The integral over the
transitory shock is split at next period's kink, so no quadrature node crosses it:
below the kink next period consumes everything and its marginal value is known in
closed form, and that piece is integrated by Gauss-Legendre in the shock's
standard-normal coordinate; above the kink next period's marginal value is
represented by cubic Hermite pieces on a fixed log-spaced grid, and the expectation
of every piece against the lognormal is a closed form in its partial moments, so
that part of the expectation is a sparse matrix built once per asset grid. The
permanent shock is integrated by Gauss-Legendre as well, and the unemployment
event, an atom in the transitory shock, has its own operator in the inverse
permanent shock. The construction is the one Christiano and Fisher (2000) describe
when they note that the images vanish under a density, and Iskhakov, Jorgensen,
Rust and Schjerning (2017, Appendix B) prescribe when they integrate once on each
side of the kink; the precomputed operator is in the spirit of Judd, Maliar, Maliar
and Tsener (2017).

What the solver covers: the IndShock problem with an artificial borrowing
constraint that binds (the kink is what the split handles), truncated lognormal
permanent and transitory shocks with positive standard deviations, and unemployment
that pays a positive income or does not occur. It builds the consumption function
with linear or cubic Hermite interpolation; it does not build the value function.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csr_matrix

from HARK.Calibration.Income.IncomeProcesses import (
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    CubicInterp,
    IndShockConsumerType,
    IndShockConsumerType_constructors_default,
    IndShockConsumerType_defaults,
    calc_human_wealth,
    calc_mpc_max,
    calc_mpc_min,
    calc_patience_factor,
    check_cubic_pairing,
)
from HARK.distributions import (
    DiscreteDistributionLabeled,
    MeanOneTruncatedLogNormal,
    TruncatedLognormal,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
    expected,
)
from HARK.interpolation import (
    LinearInterp,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA
from HARK.utilities import NullFunc

__all__ = [
    "LognormalIncomeLaw",
    "KinkSplitExpectation",
    "kink_of_solution",
    "convex_segments_above_kink",
    "make_lognormal_income_law",
    "make_income_dstn_from_law",
    "solve_one_period_ConsIndShockContinuous",
    "IndShockContinuousConsumerType",
    "init_idiosyncratic_shocks_continuous",
]

# Standard deviations of the underlying normal used in place of a missing truncation
# bound: the mass beyond eight is 1e-15.
Z_UNBOUNDED = 8.0
# The operators keep cells one standard deviation beyond each truncation bound, and
# within nine either way.
Z_OPERATOR = 9.0
# Rows of the unemployment operator whose landing base is below this have all their
# mass at the unemployment income (end-of-period assets of zero).
C_MIN = 1e-12
# A cell much narrower than its distance from zero is integrated by quadrature: the
# binomial expansion of its partial moments cancels catastrophically.
NARROW_CELL = 200.0

_INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)
_BINOM = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]
_GAUSS_LEGENDRE = {}


def _gauss_legendre(n):
    if n not in _GAUSS_LEGENDRE:
        _GAUSS_LEGENDRE[n] = leggauss(int(n))
    return _GAUSS_LEGENDRE[n]


def _phi(z):
    return _INV_SQRT2PI * np.exp(-0.5 * np.square(z))


def _z_bounds(law):
    """The interval of the underlying standard normal a truncated lognormal lives on."""
    return max(law.bot, -Z_UNBOUNDED), min(law.top, Z_UNBOUNDED)


def _z_operator_bounds(law):
    """The interval of the underlying normal whose landing points the operators cover."""
    return max(law.bot - 1.0, -Z_OPERATOR), min(law.top + 1.0, Z_OPERATOR)


def _x_of_z(law, z):
    """A truncated lognormal's value at the standard-normal coordinate z."""
    return np.exp(law.mu + law.sigma * np.asarray(z, dtype=float))


def _z_of_x(law, x):
    """The standard-normal coordinate of x; minus infinity at and below zero."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.log(x) - law.mu) / law.sigma
    return np.where(x > 0, z, -np.inf)


def _normal_weight(law, z):
    """The density of the truncated standard normal behind a truncated lognormal."""
    return _phi(z) / law.mass


def _legendre_rule(law, lo, hi, n):
    """Nodes and weights integrating against the truncated normal's density on [lo, hi]."""
    x, w = _gauss_legendre(n)
    half = 0.5 * (hi - lo)
    z = half * x + 0.5 * (hi + lo)
    return z, half * w * _normal_weight(law, z)


class LognormalIncomeLaw:
    """
    The continuous income process behind HARK's lognormal income distribution: a
    mean-one truncated lognormal permanent shock, a mean-one truncated lognormal
    transitory shock, and unemployment, which pays IncUnemp with probability
    UnempPrb; when employed, the transitory shock is scaled by
    (1 - UnempPrb * IncUnemp) / (1 - UnempPrb), so that its mean stays one. The
    standard deviations are those of the logs after truncation, as HARK's
    discretized process states them.

    Parameters
    ----------
    PermShkStd : float
        Standard deviation of the log permanent shock.
    TranShkStd : float
        Standard deviation of the log transitory shock when employed.
    UnempPrb : float
        Probability of unemployment.
    IncUnemp : float
        Transitory income when unemployed.
    PermShkTrunc : (float, float)
        Truncation of the permanent shock, in standard deviations of its
        underlying normal; None on a side is no bound there.
    TranShkTrunc : (float, float)
        Truncation of the transitory shock, likewise.
    """

    def __init__(
        self,
        PermShkStd,
        TranShkStd,
        UnempPrb=0.0,
        IncUnemp=0.0,
        PermShkTrunc=(-3.0, 3.0),
        TranShkTrunc=(-3.0, 3.0),
    ):
        if not (PermShkStd > 0.0 and TranShkStd > 0.0):
            raise ValueError(
                "The continuous income law needs positive standard deviations; a shock"
                " without dispersion is an atom, which the atoms' solver handles."
            )
        self.PermShkStd = float(PermShkStd)
        self.TranShkStd = float(TranShkStd)
        self.UnempPrb = float(UnempPrb)
        self.IncUnemp = float(IncUnemp)
        self.PermShkTrunc = tuple(PermShkTrunc)
        self.TranShkTrunc = tuple(TranShkTrunc)
        self.kappa = (
            (1.0 - self.UnempPrb * self.IncUnemp) / (1.0 - self.UnempPrb)
            if self.UnempPrb > 0.0
            else 1.0
        )
        self.PermShkLaw = MeanOneTruncatedLogNormal.from_sd_log(
            self.PermShkStd, *self.PermShkTrunc
        )
        self.TranShkLaw = MeanOneTruncatedLogNormal.from_sd_log(
            self.TranShkStd, *self.TranShkTrunc
        )
        # The transitory shock when employed, with mean kappa
        self.EmpShkLaw = TruncatedLognormal(
            self.TranShkLaw.mu + np.log(self.kappa),
            self.TranShkLaw.sigma,
            *self.TranShkTrunc,
        )
        # The inverse of the permanent shock, which the unemployment operator uses
        bot, top = self.PermShkTrunc
        self.InvPermShkLaw = TruncatedLognormal(
            -self.PermShkLaw.mu,
            self.PermShkLaw.sigma,
            None if top is None else -top,
            None if bot is None else -bot,
        )

    def key(self):
        """What distinguishes one law from another, for the operators' cache."""
        return (
            self.PermShkStd,
            self.TranShkStd,
            self.UnempPrb,
            self.IncUnemp,
            self.PermShkTrunc,
            self.TranShkTrunc,
        )

    @property
    def infimum(self):
        """The infima of the permanent shock and of transitory income."""
        emp_inf = float(self.EmpShkLaw.infimum[0])
        xi_inf = min(self.IncUnemp, emp_inf) if self.UnempPrb > 0.0 else emp_inf
        return float(self.PermShkLaw.infimum[0]), xi_inf

    @property
    def supremum(self):
        """The suprema of the permanent shock and of transitory income."""
        emp_sup = float(self.EmpShkLaw.supremum[0])
        xi_sup = max(self.IncUnemp, emp_sup) if self.UnempPrb > 0.0 else emp_sup
        return float(self.PermShkLaw.supremum[0]), xi_sup

    def worst_income_probability(self):
        """
        The probability of the worst income event: UnempPrb when unemployment pays
        nothing, since zero income is then the worst event whatever the permanent
        shock; otherwise zero, since the infima of continuous laws carry no mass.
        """
        if self.UnempPrb > 0.0 and self.IncUnemp == 0.0:
            return self.UnempPrb
        return 0.0

    def natural_borrowing_constraint(self, mNrmMinNext, Rfree, PermGroFac):
        """
        The most that can be repaid out of the worst income path: assets at which
        the worst draw lands exactly on next period's minimum. The binding permanent
        shock is its infimum when next period's minimum is below the worst transitory
        income, and its supremum otherwise.
        """
        (psi_inf, xi_inf), (psi_sup, _) = self.infimum, self.supremum
        gap = mNrmMinNext - xi_inf
        psi = psi_inf if gap <= 0.0 else psi_sup
        return gap * PermGroFac * psi / Rfree

    def discretize(self, PermShkCount, TranShkCount, seed=0):
        """
        The law's own equiprobable discretization, laid out as HARK's
        BufferStockIncShkDstn: the permanent atoms combined with the unemployment
        atom and the employed atoms.

        Parameters
        ----------
        PermShkCount : int
            Number of permanent shock atoms.
        TranShkCount : int
            Number of employed transitory shock atoms.
        seed : int, optional
            Seed of the joint distribution's random number generator.

        Returns
        -------
        DiscreteDistributionLabeled
        """
        perm = self.PermShkLaw.discretize(PermShkCount, method="equiprobable")
        tran = self.TranShkLaw.discretize(TranShkCount, method="equiprobable")
        if self.UnempPrb > 0.0:
            tran = add_discrete_outcome_constant_mean(
                tran, p=self.UnempPrb, x=self.IncUnemp
            )
        joint = combine_indep_dstns(perm, tran, seed=seed)
        return DiscreteDistributionLabeled(
            name="Joint distribution of permanent and transitory shocks to income",
            var_names=["PermShk", "TranShk"],
            pmv=joint.pmv,
            atoms=joint.atoms,
            limit=joint.limit,
            seed=seed,
        )


def kink_of_solution(solution_next):
    """
    The cash-on-hand at which next period's borrowing constraint stops binding:
    the largest knot of its unconstrained consumption function on the constrained
    line c = m - mNrmMin, else the crossing of the two found by bisection. Infinite
    when consumption equals cash-on-hand everywhere, as in the terminal period.

    Parameters
    ----------
    solution_next : ConsumerSolution
        Next period's solution.

    Returns
    -------
    float
    """
    cFunc = solution_next.cFunc
    m_min = float(solution_next.mNrmMin)

    def below_line(m):
        return float(cFunc(np.array([m]))[0]) < m - m_min

    if hasattr(cFunc, "functions") and hasattr(cFunc.functions[0], "x_list"):
        unc = cFunc.functions[0]
        x = np.asarray(unc.x_list, dtype=float)
        y = np.asarray(unc.y_list, dtype=float)
        on_line = np.flatnonzero(np.abs(y - (x - m_min)) < 1e-9 * np.maximum(1.0, np.abs(x)))
        if on_line.size:
            return float(x[on_line].max())
        lo, hi = float(x[0]), float(min(x[-1], 50.0))
    else:
        lo, hi = m_min, 50.0
    if not below_line(hi):
        return np.inf
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if below_line(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def policy_marginals(cFunc, CRRA):
    """
    A function returning next period's marginal and marginal marginal value of
    cash-on-hand from one evaluation of its consumption function and slope:
    v'(m) = c^(-rho) and v''(m) = -rho c^(-rho-1) c'(m).
    """
    rho = float(CRRA)

    def vfun(m):
        c, dcdm = cFunc.eval_with_derivative(np.asarray(m, dtype=float))
        c = np.asarray(c, dtype=float)
        v1 = c ** (-rho)
        return v1, -rho * v1 * np.asarray(dcdm, dtype=float) / c

    return vfun


def _log_grid(lo, hi, ratio):
    return np.exp(np.arange(np.log(lo), np.log(hi) + np.log(ratio), np.log(ratio)))


def _cell_weights(X, base, scale, m_lo, m_hi, w1=0.0, w2=0.0, n_gl=8):
    """
    Weights on (v_L, s_L, v_R, s_R), the values and slopes of a cubic Hermite
    piece on the cell [m_lo, m_hi], giving E[X^w1 p(m') 1{m' in cell}] and
    E[X^w2 p'(m') 1{m' in cell}] for m' = base + scale X, X a truncated lognormal.
    In X the cell is [x_lo, x_hi] and tau = (X - x_lo)/H; the moments of tau^n on
    the cell come from X's partial moments by the binomial expansion, or, on a cell
    much narrower than its distance from zero, from Gauss-Legendre against X's
    density.
    """
    base, scale = np.asarray(base, dtype=float), np.asarray(scale, dtype=float)
    m_lo, m_hi = np.asarray(m_lo, dtype=float), np.asarray(m_hi, dtype=float)
    h = m_hi - m_lo
    x_lo, x_hi, H = (m_lo - base) / scale, (m_hi - base) / scale, h / scale
    x_lo, x_hi, H, h = np.broadcast_arrays(x_lo, x_hi, H, h)
    orders = sorted({w1 + k for k in range(4)} | {w2 + k for k in range(3)})
    PM = {k: X.partial_moment(k, x_lo, x_hi) for k in orders}
    L = -x_lo
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):

        def T(w, n):
            return sum(_BINOM[n][k] * L ** (n - k) * PM[w + k] for k in range(n + 1)) / H**n

        T1 = [T(w1, n) for n in range(4)]
        T2 = [T(w2, n) for n in range(3)]
    narrow = x_lo > NARROW_CELL * H
    if np.any(narrow):
        xg, wg = _gauss_legendre(n_gl)
        lo_n, H_n = x_lo[narrow][..., None], H[narrow][..., None]
        t = 0.5 * (xg + 1.0)
        xx = lo_n + H_n * t
        dens = 0.5 * H_n * wg * X.pdf(xx)
        for lst, w in ((T1, w1), (T2, w2)):
            for n in range(len(lst)):
                arr = np.array(lst[n], dtype=float, copy=True)
                arr[narrow] = (dens * xx**w * t**n).sum(-1)
                lst[n] = arr
    T0, T1_, T2_, T3 = T1
    U0, U1, U2 = T2
    W1 = np.stack(
        [2 * T3 - 3 * T2_ + T0, h * (T3 - 2 * T2_ + T1_), -2 * T3 + 3 * T2_, h * (T3 - T2_)],
        axis=-1,
    )
    W2 = np.stack(
        [(6 * U2 - 6 * U1) / h, 3 * U2 - 4 * U1 + U0, (6 * U1 - 6 * U2) / h, 3 * U2 - 2 * U1],
        axis=-1,
    )
    return W1, W2


def _masked_cells(vg1, vg2, grid, m_kink):
    """
    Node data per cell, (v'_L, v''_L, v'_R, v''_R) flattened, with every cell at or
    below the one holding the kink zeroed; and that cell's index.
    """
    k = int(np.searchsorted(grid, m_kink, side="right")) - 1
    X = np.stack([vg1[:-1], vg2[:-1], vg1[1:], vg2[1:]], axis=1)
    if k >= 0:
        X[: k + 1] = 0.0
    return X.ravel(), k


class _EmployedOperator:
    """
    The employed branch's integral over the transitory shock above the kink, for
    fixed landing bases b = R a / (G psi): two sparse matrices acting on next
    period's marginal value and its slope at the grid's nodes.
    """

    def __init__(self, law, b, grid):
        self.law = law
        self.shape = np.asarray(b).shape
        self.b = np.asarray(b, dtype=float).ravel()
        self.grid = g = grid
        z_lo, z_hi = _z_operator_bounds(law.EmpShkLaw)
        th_lo, th_hi = float(_x_of_z(law.EmpShkLaw, z_lo)), float(_x_of_z(law.EmpShkLaw, z_hi))
        h = np.diff(g)
        nc = h.size
        rows, cols, d1, d2 = [], [], [], []
        for r, br in enumerate(self.b):
            c0 = max(int(np.searchsorted(g, br + th_lo, side="right")) - 1, 0)
            c1 = min(int(np.searchsorted(g, br + th_hi, side="left")), nc)
            cs = np.arange(c0, c1)
            if cs.size == 0:
                continue
            W1, W2 = _cell_weights(law.EmpShkLaw, 0.0, 1.0, g[cs] - br, g[cs + 1] - br)
            rows.append(np.full(4 * cs.size, r))
            cols.append((4 * cs[:, None] + np.arange(4)[None, :]).ravel())
            d1.append(W1.ravel())
            d2.append(W2.ravel())
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        self.M1 = csr_matrix((np.concatenate(d1), (rows, cols)), shape=(self.b.size, 4 * nc))
        self.M2 = csr_matrix((np.concatenate(d2), (rows, cols)), shape=(self.b.size, 4 * nc))

    def apply(self, vg1, vg2, vk2, m_kink, m_min_next, rho):
        if not np.isfinite(m_kink):
            return np.zeros(self.shape), np.zeros(self.shape)
        g = self.grid
        x, k = _masked_cells(vg1, vg2, g, m_kink)
        E1, E2 = self.M1 @ x, self.M2 @ x
        if k + 1 < g.size and g[k + 1] > m_kink:
            # The cell from the kink to the next node: its left value is the
            # constrained branch's exact marginal value
            W1, W2 = _cell_weights(
                self.law.EmpShkLaw, 0.0, 1.0, m_kink - self.b, g[k + 1] - self.b
            )
            xk = np.array([(m_kink - m_min_next) ** (-rho), vk2, vg1[k + 1], vg2[k + 1]])
            E1, E2 = E1 + W1 @ xk, E2 + W2 @ xk
        return E1.reshape(self.shape), E2.reshape(self.shape)


class _UnemployedOperator:
    """
    The unemployment branch, with the permanent shock continuous: in L = 1/psi the
    landing point is IncUnemp + c L with c = R a / G, so each cell of next period's
    marginal value is a closed form in L's partial moments of orders rho to rho + 3.
    """

    def __init__(self, law, c, grid, rho):
        self.law, self.rho, self.grid = law, float(rho), grid
        Ld = law.InvPermShkLaw
        self.c = np.asarray(c, dtype=float).ravel()
        self.zero = self.c <= C_MIN
        g, tu = grid, law.IncUnemp
        h = np.diff(g)
        nc = h.size
        z_lo, z_hi = _z_operator_bounds(Ld)
        l_lo, l_hi = float(_x_of_z(Ld, z_lo)), float(_x_of_z(Ld, z_hi))
        rows, cols, d1, d2 = [], [], [], []
        for r, cr in enumerate(self.c):
            if self.zero[r]:
                continue
            c0 = max(int(np.searchsorted(g, tu + cr * l_lo, side="right")) - 1, 0)
            c1 = min(int(np.searchsorted(g, tu + cr * l_hi, side="left")), nc)
            cs = np.arange(c0, c1)
            if cs.size == 0:
                continue
            W1, W2 = _cell_weights(Ld, tu, cr, g[cs], g[cs + 1], self.rho, self.rho + 1.0)
            rows.append(np.full(4 * cs.size, r))
            cols.append((4 * cs[:, None] + np.arange(4)[None, :]).ravel())
            d1.append(W1.ravel())
            d2.append(W2.ravel())
        if rows:
            rows, cols = np.concatenate(rows), np.concatenate(cols)
            self.M1 = csr_matrix((np.concatenate(d1), (rows, cols)), shape=(self.c.size, 4 * nc))
            self.M2 = csr_matrix((np.concatenate(d2), (rows, cols)), shape=(self.c.size, 4 * nc))
        else:
            self.M1 = self.M2 = csr_matrix((self.c.size, 4 * nc))

    def apply(self, vg1, vg2, vk2, vu1, vu2, m_kink, m_min_next, n_cnst):
        rho, tu, Ld, c, g = self.rho, self.law.IncUnemp, self.law.InvPermShkLaw, self.c, self.grid
        live = ~self.zero
        U1, U2 = np.zeros(c.size), np.zeros(c.size)
        if np.isfinite(m_kink):
            x, k = _masked_cells(vg1, vg2, g, m_kink)
            U1, U2 = self.M1 @ x, self.M2 @ x
            if k + 1 < g.size and g[k + 1] > m_kink and live.any():
                W1, W2 = _cell_weights(Ld, tu, c[live], m_kink, g[k + 1], rho, rho + 1.0)
                xk = np.array([(m_kink - m_min_next) ** (-rho), vk2, vg1[k + 1], vg2[k + 1]])
                U1[live] += W1 @ xk
                U2[live] += W2 @ xk
        # The constrained side, IncUnemp + c L below the kink, in closed form
        z_lo, z_hi = _z_bounds(Ld)
        if (not np.isfinite(m_kink)) or m_kink > tu:
            if np.isfinite(m_kink):
                with np.errstate(divide="ignore"):
                    zs = np.clip(_z_of_x(Ld, (m_kink - tu) / np.where(live, c, 1.0)), z_lo, z_hi)
            else:
                zs = np.full(c.size, z_hi)
            act = live & (zs > z_lo)
            if act.any():
                xg, wg = _gauss_legendre(n_cnst)
                z_s = zs[act][:, None]
                half = 0.5 * (z_s - z_lo)
                z = half * xg + 0.5 * (z_s + z_lo)
                w = half * wg * _normal_weight(Ld, z)
                Lq = _x_of_z(Ld, z)
                m = tu + c[act][:, None] * Lq
                v1 = (m - m_min_next) ** (-rho)
                U1[act] += (w * Lq**rho * v1).sum(-1)
                U2[act] += (w * Lq ** (rho + 1.0) * (-rho) * v1 / (m - m_min_next)).sum(-1)
        U1[self.zero] = Ld.moment(rho) * vu1
        U2[self.zero] = Ld.moment(rho + 1.0) * vu2
        return U1, U2


_OPERATORS = {}


class KinkSplitExpectation:
    """
    Next period's expected marginal value, E[psi^(-rho) v'(m')], and marginal
    marginal value, E[psi^(-rho-1) v''(m')], with m' = R a/(G psi) + theta, for a
    fixed grid of end-of-period assets, integrated against the continuous laws and
    split at next period's kink. Built once per asset grid, rates and law; each
    period then costs one evaluation of next period's consumption function at the
    operator's nodes and two sparse matrix products per branch.

    Parameters
    ----------
    law : LognormalIncomeLaw
        The income process.
    aNrm : np.array
        End-of-period assets.
    Rfree : float
        Risk free interest factor.
    PermGroFac : float
        Permanent income growth factor.
    CRRA : float
        Coefficient of relative risk aversion.
    CnstQuadCount : int
        Gauss-Legendre nodes on the constrained piece of each integral.
    PermQuadCount : int
        Gauss-Legendre nodes in the permanent shock.
    OperatorGridRatio : float
        Ratio between successive nodes of the operators' log grid.
    """

    def __init__(
        self,
        law,
        aNrm,
        Rfree,
        PermGroFac,
        CRRA,
        CnstQuadCount=16,
        PermQuadCount=16,
        OperatorGridRatio=1.02,
    ):
        self.law, self.rho = law, float(CRRA)
        self.n_cnst = int(CnstQuadCount)
        a = np.asarray(aNrm, dtype=float)
        z_lo, z_hi = _z_bounds(law.PermShkLaw)
        zq, wq = _legendre_rule(law.PermShkLaw, z_lo, z_hi, int(PermQuadCount))
        self.psi_q, self.w_q = _x_of_z(law.PermShkLaw, zq), wq
        self.b = (float(Rfree) / (float(PermGroFac) * self.psi_q[None, :])) * a[:, None]
        self.c = float(Rfree) * a / float(PermGroFac)
        emp = law.EmpShkLaw
        self.tran_z = _z_bounds(emp)
        ez_lo, ez_hi = _z_operator_bounds(emp)
        th_lo, th_hi = float(_x_of_z(emp, ez_lo)), float(_x_of_z(emp, ez_hi))
        top = float(np.max(self.b)) + th_hi
        bottom = th_lo
        if law.UnempPrb > 0.0:
            lz_lo, lz_hi = _z_operator_bounds(law.InvPermShkLaw)
            top = max(top, law.IncUnemp + float(np.max(self.c)) * float(_x_of_z(law.InvPermShkLaw, lz_hi)))
            bottom = min(bottom, law.IncUnemp)
        self.grid = _log_grid(0.5 * bottom, 1.05 * top, float(OperatorGridRatio))
        self.employed = _EmployedOperator(law, self.b, self.grid)
        self.unemployed = (
            _UnemployedOperator(law, self.c, self.grid, self.rho) if law.UnempPrb > 0.0 else None
        )

    @classmethod
    def cached(cls, law, aNrm, Rfree, PermGroFac, CRRA, CnstQuadCount, PermQuadCount, OperatorGridRatio):
        """The operator for these inputs, built on first use and kept for reuse."""
        a = np.asarray(aNrm, dtype=float)
        key = (
            law.key(),
            a.tobytes(),
            float(Rfree),
            float(PermGroFac),
            float(CRRA),
            int(CnstQuadCount),
            int(PermQuadCount),
            float(OperatorGridRatio),
        )
        op = _OPERATORS.get(key)
        if op is None:
            if len(_OPERATORS) > 64:
                _OPERATORS.clear()
            op = _OPERATORS[key] = cls(
                law, a, Rfree, PermGroFac, CRRA, CnstQuadCount, PermQuadCount, OperatorGridRatio
            )
        return op

    def _constrained_piece(self, zs, m_min_next):
        """The employed integral below the kink, where v' = (m' - mNrmMin')^(-rho)."""
        law, rho = self.law.EmpShkLaw, self.rho
        z_lo = self.tran_z[0]
        E1, E2 = np.zeros(self.b.shape), np.zeros(self.b.shape)
        act = zs > z_lo
        if act.any():
            x, wx = _gauss_legendre(self.n_cnst)
            z_s = zs[act][:, None]
            half = 0.5 * (z_s - z_lo)
            z = half * x + 0.5 * (z_s + z_lo)
            w = half * wx * _normal_weight(law, z)
            m = self.b[act][:, None] + _x_of_z(law, z) - m_min_next
            v1 = m ** (-rho)
            E1[act] = (w * v1).sum(-1)
            E2[act] = (w * (-rho) * v1 / m).sum(-1)
        return E1, E2

    def expect(self, vfun, m_kink, m_min_next):
        """
        Parameters
        ----------
        vfun : callable
            Next period's (v', v'') at an array of cash-on-hand, from
            :func:`policy_marginals`.
        m_kink : float
            Next period's kink, from :func:`kink_of_solution`.
        m_min_next : float
            Next period's minimum cash-on-hand.

        Returns
        -------
        E1, E2 : np.array
            E[psi^(-rho) v'(m')] and E[psi^(-rho-1) v''(m')] at the asset grid.
        """
        law, rho = self.law, self.rho
        g = self.grid
        ng = g.size
        finite = np.isfinite(m_kink)
        above_kink = [m_kink + 1e-10 * max(1.0, m_kink)] if finite else [g[0]]
        v1, v2 = vfun(np.concatenate([g, above_kink, [law.IncUnemp]]))
        vg1, vg2, vk2, vu1, vu2 = v1[:ng], v2[:ng], v2[ng], v1[ng + 1], v2[ng + 1]
        z_lo, z_hi = self.tran_z
        if finite:
            zs = np.clip(_z_of_x(law.EmpShkLaw, m_kink - self.b), z_lo, z_hi)
        else:
            zs = np.full(self.b.shape, z_hi)
        e1, e2 = self._constrained_piece(zs, m_min_next)
        u1, u2 = self.employed.apply(vg1, vg2, vk2, m_kink, m_min_next, rho)
        e1, e2 = e1 + u1, e2 + u2
        E1 = ((self.w_q * self.psi_q ** (-rho))[None, :] * e1).sum(1)
        E2 = ((self.w_q * self.psi_q ** (-rho - 1.0))[None, :] * e2).sum(1)
        if self.unemployed is None:
            return E1, E2
        U1, U2 = self.unemployed.apply(vg1, vg2, vk2, vu1, vu2, m_kink, m_min_next, self.n_cnst)
        u = law.UnempPrb
        return (1.0 - u) * E1 + u * U1, (1.0 - u) * E2 + u * U2


def convex_segments_above_kink(solution, tol=1e-9):
    """
    How many segments of a cubic consumption function violate concavity above
    this period's kink, where the lower envelope uses the cubic rather than the
    constrained line. On each segment the cubic Hermite's second derivative is
    linear, so it is positive somewhere iff it is positive at an end, which is
    computed exactly from the knots' values and slopes. Linear interpolation is
    concave iff its chord slopes never rise, which is what is counted for it.

    Parameters
    ----------
    solution : ConsumerSolution
        A solved period, whose cFunc is a LowerEnvelope of the unconstrained
        function and the constrained line.
    tol : float
        Allowance for rounding.

    Returns
    -------
    segments : int
        Number of segments wholly or partly above the kink.
    convex : int
        Number of those where the function violates concavity.
    """
    unc = solution.cFunc.functions[0]
    m, c = np.asarray(unc.x_list, dtype=float), np.asarray(unc.y_list, dtype=float)
    m_kink = kink_of_solution(solution)
    above = m[1:] > m_kink
    h, slope = np.diff(m), np.diff(c) / np.diff(m)
    if not hasattr(unc, "dydx_list"):
        rises = np.diff(slope)[above[1:]] > tol
        return int(above.sum()), int(rises.sum())
    s = np.asarray(unc.dydx_list, dtype=float)
    c2 = np.maximum(
        (6 * slope - 4 * s[:-1] - 2 * s[1:]) / h,
        (-6 * slope + 2 * s[:-1] + 4 * s[1:]) / h,
    )
    return int(above.sum()), int(np.sum((c2 > tol) & above))


def solve_one_period_ConsIndShockContinuous(
    solution_next,
    IncShkDstn,
    IncShkLaw,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
    CnstQuadCount=16,
    PermQuadCount=16,
    OperatorGridRatio=1.02,
):
    """
    Solves one period of the IndShock problem with the expectation over the income
    shocks integrated against their continuous laws, split at next period's kink.
    The end-of-period asset grid starts at the artificial borrowing constraint,
    which must bind, and includes it, so that a knot sits exactly on this period's
    kink and the unconstrained function is used only above it.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : DiscreteDistribution
        The law's own discretization, used for the expected income and for the
        objects that still read atoms.
    IncShkLaw : LognormalIncomeLaw
        The continuous income process.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of the
        succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt : float
        Artificial borrowing constraint, which must exceed the natural one.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values, above the constraint.
    vFuncBool : boolean
        Must be False: the value function is not built.
    CubicBool : boolean
        An indicator for whether the solver should use cubic or linear interpolation.
    CnstQuadCount : int
        Gauss-Legendre nodes on the constrained piece of each integral.
    PermQuadCount : int
        Gauss-Legendre nodes in the permanent shock.
    OperatorGridRatio : float
        Ratio between successive nodes of the operators' log grid.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem.
    """
    # Cubic Hermite interpolation is the pairing the continuous expectation allows
    check_cubic_pairing(CubicBool, continuous=True)
    if vFuncBool:
        raise NotImplementedError(
            "The continuous-expectation solver does not build the value function;"
            " set vFuncBool=False."
        )
    law = IncShkLaw
    if law.UnempPrb > 0.0 and law.IncUnemp <= 0.0:
        raise NotImplementedError(
            "The continuous-expectation solver needs unemployment to pay a positive"
            " income: with zero income the natural borrowing constraint is zero, and"
            " the constraint does not bind."
        )
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb

    Ex_IncNext = expected(lambda x: x["PermShk"] * x["TranShk"], IncShkDstn)
    hNrmNow = calc_human_wealth(solution_next.hNrm, PermGroFac, Rfree, Ex_IncNext)
    mNrmMinNext = float(solution_next.mNrmMin)
    BoroCnstNat = law.natural_borrowing_constraint(mNrmMinNext, Rfree, PermGroFac)
    if BoroCnstArt is None or not BoroCnstArt > BoroCnstNat:
        raise ValueError(
            "The continuous-expectation solver needs an artificial borrowing"
            f" constraint that binds: BoroCnstArt is {BoroCnstArt!r} against the laws'"
            f" natural constraint {BoroCnstNat:.6f}."
        )
    mNrmMinNow = float(BoroCnstArt)

    # The bounding MPCs and the PDV of human wealth. The worst income event has
    # probability zero under continuous laws, so the MPC at the natural constraint
    # is one; at the binding artificial constraint it is one in any case.
    WorstIncPrb = law.worst_income_probability()
    PatFac = calc_patience_factor(Rfree, DiscFacEff, CRRA)
    MPCminNow = calc_mpc_min(solution_next.MPCmin, PatFac)
    MPCmaxUnc = calc_mpc_max(
        solution_next.MPCmax, WorstIncPrb, CRRA, PatFac, BoroCnstNat, BoroCnstArt
    )
    MPCmaxNow = 1.0
    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

    # The borrowing-constrained consumption function
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]), np.array([0.0, 1.0])
    )

    # End-of-period assets, from the constraint up, including the constraint itself:
    # the point it maps to is this period's kink
    aNrmNow = np.concatenate(([mNrmMinNow], mNrmMinNow + np.asarray(aXtraGrid)))

    # The expectation over the continuous shocks, split at next period's kink
    expectation = KinkSplitExpectation.cached(
        law, aNrmNow, Rfree, PermGroFac, CRRA, CnstQuadCount, PermQuadCount, OperatorGridRatio
    )
    E1, E2 = expectation.expect(
        policy_marginals(solution_next.cFunc, CRRA), kink_of_solution(solution_next), mNrmMinNext
    )
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * E1

    # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
    cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    mNrmNow = cNrmNow + aNrmNow

    # Limiting consumption is zero as m approaches the natural constraint
    c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
    m_for_interpolation = np.insert(mNrmNow, 0, BoroCnstNat)

    if CubicBool:
        vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
        EndOfPrdvPP = vPPfacEff * E2
        dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
        MPC = dcda / (dcda + 1.0)
        MPC_for_interpolation = np.insert(MPC, 0, MPCmaxUnc)
        cFuncNowUnc = CubicInterp(
            m_for_interpolation,
            c_for_interpolation,
            MPC_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )
    else:
        cFuncNowUnc = LinearInterp(
            m_for_interpolation, c_for_interpolation, cFuncLimitIntercept, cFuncLimitSlope
        )

    cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst, nan_bool=False)
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)
    vPPfuncNow = MargMargValueFuncCRRA(cFuncNow, CRRA) if CubicBool else NullFunc()
    return ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=NullFunc(),
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxNow,
    )


def make_lognormal_income_law(
    PermShkStd, TranShkStd, UnempPrb, IncUnemp, PermShkTrunc, TranShkTrunc, T_cycle
):
    """
    Constructs the continuous income law of each period from the parameters of
    HARK's lognormal income process.

    Parameters
    ----------
    PermShkStd : [float]
        Standard deviation of the log permanent shock, by period.
    TranShkStd : [float]
        Standard deviation of the log transitory shock, by period.
    UnempPrb : float
        Probability of unemployment.
    IncUnemp : float
        Transitory income when unemployed.
    PermShkTrunc : (float, float)
        Truncation of the permanent shock, in standard deviations.
    TranShkTrunc : (float, float)
        Truncation of the transitory shock, in standard deviations.
    T_cycle : int
        Number of periods in the agent's cycle.

    Returns
    -------
    IncShkLaw : [LognormalIncomeLaw]
    """
    if len(PermShkStd) != T_cycle or len(TranShkStd) != T_cycle:
        raise ValueError("PermShkStd and TranShkStd must have T_cycle entries.")
    return [
        LognormalIncomeLaw(
            PermShkStd[t], TranShkStd[t], UnempPrb, IncUnemp, PermShkTrunc, TranShkTrunc
        )
        for t in range(T_cycle)
    ]


def make_income_dstn_from_law(IncShkLaw, PermShkCount, TranShkCount, RNG):
    """
    Constructs each period's discretized income distribution from its continuous
    law, so that the atoms the simulator and the transition matrices use are
    those of the process the solver integrates.

    Parameters
    ----------
    IncShkLaw : [LognormalIncomeLaw]
        The continuous income law of each period.
    PermShkCount : int
        Number of permanent shock atoms.
    TranShkCount : int
        Number of employed transitory shock atoms.
    RNG : np.random.Generator
        The agent's random number generator, which seeds the distributions.

    Returns
    -------
    IncShkDstn : [DiscreteDistributionLabeled]
    """
    return [
        law.discretize(PermShkCount, TranShkCount, seed=int(RNG.integers(0, 2**31 - 1)))
        for law in IncShkLaw
    ]


IndShockContinuousConsumerType_constructors_default = (
    IndShockConsumerType_constructors_default.copy()
)
IndShockContinuousConsumerType_constructors_default.update(
    {
        "IncShkLaw": make_lognormal_income_law,
        "IncShkDstn": make_income_dstn_from_law,
        "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
        "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    }
)

IndShockContinuousConsumerType_defaults = IndShockConsumerType_defaults.copy()
IndShockContinuousConsumerType_defaults.update(
    {
        "constructors": IndShockContinuousConsumerType_constructors_default,
        "PermShkTrunc": (-3.0, 3.0),  # Truncation of the permanent shock, in standard deviations
        "TranShkTrunc": (-3.0, 3.0),  # Truncation of the transitory shock, in standard deviations
        "CnstQuadCount": 16,  # Gauss-Legendre nodes on the constrained piece of each integral
        "PermQuadCount": 16,  # Gauss-Legendre nodes in the permanent shock
        "OperatorGridRatio": 1.02,  # Ratio between successive nodes of the operators' log grid
    }
)
init_idiosyncratic_shocks_continuous = IndShockContinuousConsumerType_defaults


class IndShockContinuousConsumerType(IndShockConsumerType):
    r"""
    An IndShockConsumerType whose solver integrates the income shocks against their
    continuous laws rather than summing over discrete atoms. The laws are the
    truncated lognormals behind HARK's income process, built from the same
    parameters, and the agent's discretized distributions are those laws' own
    equiprobable atoms. Its artificial borrowing constraint must bind, unemployment
    must pay a positive income or not occur, and the value function is not built.
    With CubicBool the consumption function is concave above the kink, which the
    atoms' solver cannot promise.

    Parameters
    ----------
    PermShkTrunc : (float, float)
        Truncation of the permanent shock, in standard deviations of its underlying
        normal; None on a side is no bound there.
    TranShkTrunc : (float, float)
        Truncation of the transitory shock, likewise.
    CnstQuadCount : int
        Gauss-Legendre nodes on the constrained piece of each integral.
    PermQuadCount : int
        Gauss-Legendre nodes in the permanent shock.
    OperatorGridRatio : float
        Ratio between successive nodes of the operators' log grid.
    """

    default_ = {
        "params": IndShockContinuousConsumerType_defaults,
        "solver": solve_one_period_ConsIndShockContinuous,
        "model": "ConsIndShock.yaml",
        "track_vars": ["aNrm", "cNrm", "mNrm", "pLvl"],
    }
    time_vary_ = IndShockConsumerType.time_vary_ + ["IncShkLaw"]
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "CnstQuadCount",
        "PermQuadCount",
        "OperatorGridRatio",
    ]
