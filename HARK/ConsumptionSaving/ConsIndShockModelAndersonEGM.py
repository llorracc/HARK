"""Anderson-accelerated Endogenous Grid Method (EGM) for ``ConsIndShockModel``.

This module adds a **strictly opt-in** Anderson-accelerated EGM solver for the
infinite-horizon ``IndShockConsumerType`` consumption-saving problem.  It does
**not** change any existing behavior: the default solver and the default solution
path are untouched.  Opt in via the :class:`IndShockConsumerTypeAndersonEGM`
subclass (``cycles=0`` only).

Why
---
Plain EGM is a fixed-point iteration ``c_{k+1} = G(c_k)`` whose contraction rate
``rho(G) -> 1`` as the growth-impatience (GIC) condition approaches its edge
(``GPF -> 1``).  At ``GPF`` ~ 0.999 EGM needs ~1000+ sweeps.  Anderson
acceleration (a.k.a. Anderson mixing / DIIS) is a black-box accelerator for any
such map: it keeps the last ``depth`` iterates and residuals and forms the next
iterate as the residual-minimizing affine combination (a tiny least-squares
problem), turning the linear contraction near-superlinear -- *without ever
forming a Jacobian*.  The fixed point is identical to plain EGM, so the answer
is unchanged; only the iteration count falls.

The design decision (mix ``consumed(a)``, not ``consumption(m)``)
----------------------------------------------------------------
EGM's natural output lives on an *endogenous*, solution-dependent ``m``-grid that
moves every iteration, so mixing ``consumption(m)`` would inject resampling error
into the residual.  Instead we run Anderson on the ``consumed(a)`` vector -- the
``c`` at each node of the **invariant** post-decision asset grid ``aNrmNow``
(``= aXtraGrid + BoroCnstNat``), exactly the quantity EGM produces by inverting
the first-order condition ``cNrm = uP_inv(EndOfPrdvP)``.  ``consumed(a)`` lives on
a fixed grid and is the smooth unconstrained branch (the borrowing constraint is
applied only when assembling ``cFunc(m)``), giving Anderson an exact fixed vector
space.  The fixed point is identical to plain EGM, so the parity bar is exact.

Licence
-------
Pure public EGM (the published HARK algorithm) plus a generic public accelerator
(Anderson mixing).  No third-party / arbitrage-Jacobian code.  Cleanly PR-able to
``econ-ark/HARK`` independently of any licence-gated solver work.
"""

import numpy as np

from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    calc_boro_const_nat,
    calc_human_wealth,
    calc_m_nrm_min,
    calc_mpc_max,
    calc_mpc_min,
    calc_patience_factor,
    calc_vp_next,
    calc_vpp_next,
    calc_worst_inc_prob,
    solve_one_period_ConsIndShock,
)
from HARK.distributions import expected
from HARK.interpolation import (
    LinearInterp,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.interpolation import CubicHermiteInterp as CubicInterp
from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA

__all__ = [
    "anderson_accelerate",
    "solve_stationary_AndersonEGM",
    "IndShockConsumerTypeAndersonEGM",
]


def anderson_accelerate(sweep, x0, *, depth=5, tol=1e-10, maxit=10000,
                        lo=None, hi=None, safeguard=True, verbose=False):
    r"""Anderson-accelerate a fixed-point map ``sweep: x -> x'`` (1-D vector).

    Mixes the last ``depth`` residuals ``f_k = sweep(x_k) - x_k`` via a small
    least-squares to extrapolate the fixed point.  Convergence is the scale-free
    relative move ``max|f| / (1 + |x|)``.

    Parameters
    ----------
    sweep : callable
        ``x (ndarray) -> x' (ndarray)``; the fixed-point operator.
    x0 : ndarray
        Initial iterate.
    depth : int
        Anderson memory depth (number of past residual differences mixed).
    tol : float
        Convergence threshold on the scale-free relative move.
    maxit : int
        Maximum number of sweeps.
    lo, hi : float or ndarray, optional
        Element-wise clip bounds applied to every iterate (kept feasible).
    safeguard : bool
        If True, when a mix produces a non-finite or larger-residual iterate,
        fall back to the plain-Picard update ``x' = sweep(x)`` for that step.
        Robust on near-singular history windows / non-contractive transients;
        does not change the fixed point.
    verbose : bool
        Print per-step diagnostics.

    Returns
    -------
    x : ndarray
        Final iterate (the converged fixed point if ``converged``).
    info : dict
        ``{"iters", "converged", "move_rel", "n_fallback"}``.
    """
    x = np.array(x0, dtype=float).ravel()

    def _clip(v):
        if lo is not None or hi is not None:
            v = np.clip(v, lo, hi)
        return v

    def _move(f_vec, x_vec):
        return float(np.max(np.abs(f_vec) / (1.0 + np.abs(x_vec))))

    def _safe_sweep(xv):
        """Evaluate the sweep; return None if it raises or is non-finite."""
        try:
            out = np.asarray(sweep(xv), dtype=float).ravel()
        except (ValueError, FloatingPointError):
            return None
        return out if np.all(np.isfinite(out)) else None

    g_hist, r_hist = [], []
    n_fallback = 0
    move_rel = np.inf
    x_good = x.copy()
    for k in range(maxit):
        Gx = _safe_sweep(x)
        if Gx is None:
            # Current iterate is ill-posed (a prior mix overshot): restart from
            # the last known-good iterate with a clean Picard step.
            g_hist, r_hist = [], []
            n_fallback += 1
            Gx = _safe_sweep(x_good)
            if Gx is None:
                break  # cannot recover
            x = _clip(Gx)
            x_good = x.copy()
            continue
        x_good = x.copy()
        f = Gx - x
        move_rel = _move(f, x)
        if verbose and (k < 3 or k % 25 == 0):
            print(f"  AndersonEGM it {k + 1:5d} move_rel={move_rel:.3e}")
        if move_rel < tol:
            return _clip(Gx), {
                "iters": k + 1, "converged": True,
                "move_rel": move_rel, "n_fallback": n_fallback,
            }

        g_hist.append(Gx)
        r_hist.append(f)
        if len(r_hist) > depth + 1:
            g_hist.pop(0)
            r_hist.pop(0)

        mk = len(r_hist) - 1
        if mk == 0:
            x = _clip(Gx)  # first step: plain Picard
            continue

        dR = np.column_stack([r_hist[i + 1] - r_hist[i] for i in range(mk)])
        dG = np.column_stack([g_hist[i + 1] - g_hist[i] for i in range(mk)])
        x_and = None
        if np.all(np.isfinite(dR)) and np.all(np.isfinite(f)):
            try:
                gamma, *_ = np.linalg.lstsq(dR, f, rcond=None)
                cand = _clip(Gx - dG @ gamma)
                if np.all(np.isfinite(cand)):
                    x_and = cand
            except np.linalg.LinAlgError:
                x_and = None

        if x_and is None:
            x = _clip(Gx)  # singular/non-finite window -> Picard
            n_fallback += 1
        elif safeguard:
            try:
                f_and = np.asarray(sweep(x_and), dtype=float).ravel() - x_and
                ok = np.all(np.isfinite(f_and)) and _move(f_and, x_and) <= move_rel
            except (ValueError, FloatingPointError):
                ok = False  # mix made the (e.g. cubic) sweep ill-posed
            if ok:
                x = x_and
            else:
                x = _clip(Gx)  # mix would raise/worsen residual -> plain EGM sweep
                n_fallback += 1
        else:
            x = x_and

    return _clip(x), {
        "iters": maxit, "converged": False,
        "move_rel": move_rel, "n_fallback": n_fallback,
    }


def _converge_stationary_scalars(IncShkDstn, LivPrb, DiscFac, CRRA, Rfree,
                                 PermGroFac, BoroCnstArt, *, tol=1e-14,
                                 maxit=1_000_000):
    """Iterate the EGM scalar recursions (hNrm, MPCmin, MPCmax, mNrmMin,
    BoroCnstNat) to their infinite-horizon fixed points.

    These are exactly the per-period updates inside
    ``solve_one_period_ConsIndShock``; driving them to their (seed-independent)
    fixed points lets the policy iteration hold them constant, so the canonical
    final solution reproduces stock EGM's converged scalars exactly.  Each is a
    cheap scalar recursion.
    """
    DiscFacEff = DiscFac * LivPrb
    WorstIncPrb = calc_worst_inc_prob(IncShkDstn)
    Ex_IncNext = expected(lambda x: x["PermShk"] * x["TranShk"], IncShkDstn)
    PatFac = calc_patience_factor(Rfree, DiscFacEff, CRRA)

    h = 0.0
    mpc_min = 1.0
    mpc_max = 1.0
    m_min_next = 0.0
    for _ in range(maxit):
        h_new = calc_human_wealth(h, PermGroFac, Rfree, Ex_IncNext)
        mpc_min_new = calc_mpc_min(mpc_min, PatFac)
        boro_nat = calc_boro_const_nat(m_min_next, IncShkDstn, Rfree, PermGroFac)
        m_min_new = calc_m_nrm_min(BoroCnstArt, boro_nat)
        mpc_max_unc = calc_mpc_max(
            mpc_max, WorstIncPrb, CRRA, PatFac, boro_nat, BoroCnstArt
        )
        mpc_max_new = 1.0 if boro_nat < m_min_new else mpc_max_unc

        deltas = (
            abs(h_new - h),
            abs(mpc_min_new - mpc_min),
            abs(mpc_max_new - mpc_max),
            abs(m_min_new - m_min_next),
        )
        h, mpc_min, mpc_max, m_min_next = h_new, mpc_min_new, mpc_max_new, m_min_new
        if max(deltas) < tol:
            break

    boro_nat = calc_boro_const_nat(m_min_next, IncShkDstn, Rfree, PermGroFac)
    m_min = calc_m_nrm_min(BoroCnstArt, boro_nat)
    mpc_max_unc = calc_mpc_max(mpc_max, WorstIncPrb, CRRA, PatFac, boro_nat, BoroCnstArt)
    return {
        "hNrm": h,
        "MPCmin": mpc_min,
        "MPCmax": mpc_max,
        "MPCmaxUnc": mpc_max_unc,
        "mNrmMin": m_min,
        "BoroCnstNat": boro_nat,
        "DiscFacEff": DiscFacEff,
        "PatFac": PatFac,
    }


def solve_stationary_AndersonEGM(
    IncShkDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac, BoroCnstArt,
    aXtraGrid, vFuncBool, CubicBool, *,
    accelerate=True, anderson_depth=5, tol=1e-11, maxit=20000,
    safeguard=True, c_init=None, verbose=False,
):
    """Solve the stationary (infinite-horizon) ``ConsIndShock`` problem by
    Anderson-accelerated EGM and return the canonical ``ConsumerSolution``.

    The EGM fixed-point map ``G`` acts on ``consumed(a)`` (consumption at each
    node of the invariant post-decision asset grid).  Set ``accelerate=False``
    for plain Picard (reproduces stock EGM bit-for-bit).  Scalars are
    pre-converged; once ``consumed(a)`` converges, the canonical solution is
    produced by a single call to the stock per-period solver at the fixed point
    (so every ``ConsumerSolution`` field matches stock EGM exactly).

    Returns ``(solution_now, info)`` where ``info`` carries iteration stats.
    """
    uFunc = UtilityFuncCRRA(CRRA)
    sc = _converge_stationary_scalars(
        IncShkDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac, BoroCnstArt
    )
    DiscFacEff = sc["DiscFacEff"]
    hNrm = sc["hNrm"]
    MPCmin = sc["MPCmin"]
    MPCmaxUnc = sc["MPCmaxUnc"]
    BoroCnstNat = sc["BoroCnstNat"]
    mNrmMin = sc["mNrmMin"]

    aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat
    J = aNrmNow.size
    cFuncLimitIntercept = MPCmin * hNrm
    cFuncLimitSlope = MPCmin
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMin, mNrmMin + 1.0]), np.array([0.0, 1.0])
    )
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)

    def _build_cfunc(consumed_a, mpc, cubic):
        c_for = np.insert(consumed_a, 0, 0.0)
        m_for = np.insert(consumed_a + aNrmNow, 0, BoroCnstNat)
        if cubic and np.any(np.diff(m_for) <= 0.0):
            # Transient guard: an off-fixed-point iterate can make the endogenous
            # grid m = a + c non-monotone, which the cubic spline rejects.  Nudge
            # to strictly increasing.  Inactive at the fixed point (m is strictly
            # increasing there), so it never perturbs the converged solution.
            for i in range(1, m_for.size):
                if m_for[i] <= m_for[i - 1]:
                    m_for[i] = m_for[i - 1] + 1e-12
        if cubic:
            mpc_for = np.insert(mpc, 0, MPCmaxUnc)
            cFuncUnc = CubicInterp(
                m_for, c_for, mpc_for, cFuncLimitIntercept, cFuncLimitSlope
            )
        else:
            cFuncUnc = LinearInterp(
                m_for, c_for, cFuncLimitIntercept, cFuncLimitSlope
            )
        return LowerEnvelope(cFuncUnc, cFuncNowCnst, nan_bool=False)

    def _egm_core(consumed_a, mpc, cubic, want_mpc):
        """One EGM step on ``consumed(a)`` with a GIVEN continuation MPC (used only
        for the cubic continuation cFunc).  Returns ``(consumed(a)', mpc' | None)``.
        Holding ``mpc`` fixed makes this a *pure* function of ``consumed(a)`` -- the
        key to robust Anderson mixing in the cubic case."""
        cFunc = _build_cfunc(consumed_a, mpc, cubic)
        vPfunc = MargValueFuncCRRA(cFunc, CRRA)
        EndOfPrdvP = vPfacEff * expected(
            calc_vp_next, IncShkDstn,
            args=(aNrmNow, Rfree, CRRA, PermGroFac, vPfunc),
        )
        cNext = uFunc.derinv(EndOfPrdvP, order=(1, 0))
        mpc_out = None
        if want_mpc:
            vPPfunc = MargMargValueFuncCRRA(cFunc, CRRA)
            EndOfPrdvPP = vPPfacEff * expected(
                calc_vpp_next, IncShkDstn,
                args=(aNrmNow, Rfree, CRRA, PermGroFac, vPPfunc),
            )
            dcda = EndOfPrdvPP / uFunc.der(np.asarray(cNext), order=2)
            mpc_out = dcda / (dcda + 1.0)
        return cNext, mpc_out

    def _iterate(sweep, c0):
        """Anderson (or plain Picard) iteration of a PURE sweep on consumed(a)."""
        if accelerate:
            return anderson_accelerate(
                sweep, c0, depth=anderson_depth, tol=tol, maxit=maxit,
                lo=1e-10, hi=None, safeguard=safeguard, verbose=verbose,
            )
        c = np.maximum(c0, 1e-10)
        move_rel = np.inf
        info = {"iters": maxit, "converged": False,
                "move_rel": move_rel, "n_fallback": 0}
        for k in range(maxit):
            c_new = np.maximum(sweep(c), 1e-10)
            move_rel = float(np.max(np.abs(c_new - c) / (1.0 + np.abs(c))))
            c = c_new
            if move_rel < tol:
                info = {"iters": k + 1, "converged": True,
                        "move_rel": move_rel, "n_fallback": 0}
                break
        return c, info

    def _solve_policy(cubic, c0):
        """Solve for the stationary policy ``consumed(a)`` (and its MPC, for cubic).

        Linear: Anderson/Picard on the pure EGM sweep of ``consumed(a)``.  Cubic:
        Anderson/Picard on the JOINT state ``[consumed(a), mpc]`` (mpc enters the
        cubic continuation cFunc), so the map is a pure function of the iterate and
        convergence requires BOTH to settle -- this is exactly stock cubic EGM's
        joint contraction, just accelerated.  Returns ``(consumed_a*, mpc*, info)``.
        """
        if not cubic:
            sweep = lambda c: _egm_core(c, None, False, False)[0]
            c_star, info = _iterate(sweep, c0)
            return c_star, None, info

        # Seed the MPC consistently with the (warm-start) policy.
        _, mpc_seed = _egm_core(c0, np.full(J, MPCmaxUnc), True, True)
        if mpc_seed is None or not np.all(np.isfinite(mpc_seed)):
            mpc_seed = np.full(J, MPCmaxUnc)

        def _joint_sweep(x):
            c_in, mpc_in = x[:J], x[J:]
            c_out, mpc_out = _egm_core(c_in, mpc_in, True, True)
            return np.concatenate([c_out, mpc_out])

        lo = np.concatenate([np.full(J, 1e-10), np.full(J, 1e-6)])
        if accelerate:
            x_star, info = anderson_accelerate(
                _joint_sweep, np.concatenate([c0, mpc_seed]),
                depth=anderson_depth, tol=tol, maxit=maxit,
                lo=lo, hi=None, safeguard=safeguard, verbose=verbose,
            )
        else:
            x = np.concatenate([c0, mpc_seed])
            move_rel, info = np.inf, {"iters": maxit, "converged": False,
                                      "move_rel": np.inf, "n_fallback": 0}
            for k in range(maxit):
                x_new = np.maximum(_joint_sweep(x), lo)
                move_rel = float(np.max(np.abs(x_new - x) / (1.0 + np.abs(x))))
                x = x_new
                if move_rel < tol:
                    info = {"iters": k + 1, "converged": True,
                            "move_rel": move_rel, "n_fallback": 0}
                    break
        return x_star[:J] if accelerate else x[:J], \
            (x_star[J:] if accelerate else x[J:]), info

    # Seed: perfect-foresight consumption on the fixed grid (c = MPCmin*(m+hNrm)).
    # Seed-independent fixed point, so this never changes the answer -- it only
    # sets the starting point.  consumed(a) has NO c<=a upper bound (the borrowing
    # constraint c<=m is applied only when assembling cFunc); enforce positivity.
    if c_init is not None:
        c0 = np.maximum(np.array(c_init, dtype=float), 1e-10)
    else:
        c0 = np.maximum(MPCmin * (aNrmNow + hNrm), 1e-10)

    if CubicBool and c_init is None:
        # Warm-start the cubic solve from the converged LINEAR policy: a valid
        # converged policy gives a strictly increasing endogenous m = a + c, which
        # the cubic spline requires (an arbitrary seed can be non-monotone).
        c_lin, _, info_lin = _solve_policy(False, c0)
        c0 = c_lin
    else:
        info_lin = None

    c_star, mpc_star, info = _solve_policy(CubicBool, c0)
    if info_lin is not None:
        info = {**info, "iters_linear_warmstart": info_lin["iters"]}
    if mpc_star is None:
        mpc_star = np.full(J, MPCmaxUnc)

    # Build the converged continuation solution, then produce the canonical
    # ConsumerSolution via ONE stock per-period solve at the fixed point: every
    # field (cFunc/vPfunc/vPPfunc/vFunc/scalars) is then constructed by HARK's
    # own code, identical to plain EGM's converged output.  When vFuncBool, the
    # value function has its own fixed point, so iterate the stock solver to
    # convergence on the (already-fixed) policy.
    cFunc_star = _build_cfunc(c_star, mpc_star, CubicBool)
    sol_next = _ConsumerSolutionLite(
        cFunc=cFunc_star,
        vPfunc=MargValueFuncCRRA(cFunc_star, CRRA),
        vPPfunc=(MargMargValueFuncCRRA(cFunc_star, CRRA) if CubicBool else NullFunc()),
        # Finite value-function seed (HARK's terminal-value pattern). Needed only
        # when vFuncBool; iterated to its fixed point below on the fixed policy.
        vFunc=(ValueFuncCRRA(cFunc_star, CRRA) if vFuncBool else NullFunc()),
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        MPCmax=sc["MPCmax"],
    )

    solution_now = solve_one_period_ConsIndShock(
        sol_next, IncShkDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac,
        BoroCnstArt, aXtraGrid, vFuncBool, CubicBool,
    )

    if vFuncBool:
        # Converge the value function on the fixed policy (Anderson on the
        # value-inverse vector would also work; plain iteration is exact and the
        # value contraction is typically far from the GIC edge's policy rate).
        prev = solution_now
        for _ in range(maxit):
            cur = solve_one_period_ConsIndShock(
                prev, IncShkDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac,
                BoroCnstArt, aXtraGrid, vFuncBool, CubicBool,
            )
            m_chk = mNrmMin + aXtraGrid
            dv = np.max(np.abs(cur.vFunc(m_chk) - prev.vFunc(m_chk)))
            prev = cur
            if dv < tol:
                break
        solution_now = prev

    return solution_now, info


class _ConsumerSolutionLite(MetricObject):
    """Minimal stand-in carrying just the fields ``solve_one_period_ConsIndShock``
    reads from ``solution_next`` (cFunc, vPfunc, vPPfunc, vFunc, mNrmMin, hNrm,
    MPCmin, MPCmax).  Used to feed the converged continuation into the stock
    per-period solver for the canonical final solution."""

    def __init__(self, cFunc, vPfunc, vPPfunc, vFunc, mNrmMin, hNrm, MPCmin, MPCmax):
        self.cFunc = cFunc
        self.vPfunc = vPfunc
        self.vPPfunc = vPPfunc
        self.vFunc = vFunc
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax


class IndShockConsumerTypeAndersonEGM(IndShockConsumerType):
    """``IndShockConsumerType`` whose infinite-horizon solve uses
    Anderson-accelerated EGM.  **Opt-in and infinite-horizon only.**

    Identical model and results to ``IndShockConsumerType`` (same EGM fixed
    point); only the solution *method* differs, reaching that fixed point in far
    fewer sweeps at the growth-impatience edge.  ``cycles`` must be ``0`` (the
    accelerator targets the stationary fixed point); any other value raises
    ``NotImplementedError``.

    Extra keyword parameters (all optional): ``anderson_depth`` (5),
    ``anderson_tol`` (1e-11), ``anderson_maxit`` (20000), ``anderson_safeguard``
    (True), ``anderson_accelerate`` (True; set False for plain Picard).
    """

    def __init__(self, **kwds):
        params = {
            "anderson_depth": 5,
            "anderson_tol": 1e-11,
            "anderson_maxit": 20000,
            "anderson_safeguard": True,
            "anderson_accelerate": True,
        }
        params.update(kwds)
        self._anderson_cfg = {
            "depth": params.pop("anderson_depth"),
            "tol": params.pop("anderson_tol"),
            "maxit": params.pop("anderson_maxit"),
            "safeguard": params.pop("anderson_safeguard"),
            "accelerate": params.pop("anderson_accelerate"),
        }
        super().__init__(**params)
        self.solve_one_period = solve_one_period_ConsIndShock
        self.last_anderson_info = None

    def solve(self, verbose=False):
        """Solve the stationary problem via Anderson-accelerated EGM."""
        if self.cycles != 0:
            raise NotImplementedError(
                "IndShockConsumerTypeAndersonEGM solves the infinite-horizon "
                "(cycles=0) stationary problem only; got cycles="
                f"{self.cycles!r}."
            )
        self.pre_solve()
        time_inv = self._unpack_time_inv()
        cfg = self._anderson_cfg
        solution, info = solve_stationary_AndersonEGM(
            time_inv["IncShkDstn"], time_inv["LivPrb"], time_inv["DiscFac"],
            time_inv["CRRA"], time_inv["Rfree"], time_inv["PermGroFac"],
            time_inv["BoroCnstArt"], time_inv["aXtraGrid"],
            time_inv["vFuncBool"], time_inv["CubicBool"],
            accelerate=cfg["accelerate"], anderson_depth=cfg["depth"],
            tol=cfg["tol"], maxit=cfg["maxit"], safeguard=cfg["safeguard"],
            verbose=verbose,
        )
        self.last_anderson_info = info
        self.solution = [solution]
        return self.solution

    def _unpack_time_inv(self):
        """Pull the per-period scalars/arrays the solver needs from this agent,
        normalizing time-varying lists to their (single) stationary entry."""

        def _scalar(name):
            val = getattr(self, name)
            if isinstance(val, (list, tuple, np.ndarray)):
                return val[0]
            return val

        return {
            "IncShkDstn": self.IncShkDstn[0],
            "LivPrb": _scalar("LivPrb"),
            "DiscFac": _scalar("DiscFac"),
            "CRRA": self.CRRA,
            "Rfree": _scalar("Rfree"),
            "PermGroFac": _scalar("PermGroFac"),
            "BoroCnstArt": self.BoroCnstArt,
            "aXtraGrid": self.aXtraGrid,
            "vFuncBool": self.vFuncBool,
            "CubicBool": self.CubicBool,
        }
