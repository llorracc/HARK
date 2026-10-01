# Requirements: a truncated continuous lognormal distribution for HARK

- **Written**: 2026-10-01, by Claude (Opus 5.5) for Christopher Carroll
- **Status**: requirements for evaluation and implementation; no code yet
- **Base**: `econ-ark/HARK` main at `d741d378` (2026-09-14); every line number below refers to that commit
- **Audience**: an AI or human implementer who will first evaluate this document against the code, then build the feature

## 0. How to use this document

1. **Evaluate before building.** Check every factual claim in section 2 against the current code, since line numbers drift.
   Report anything that no longer holds before writing code.
2. **Respect the three levels.** MUST items are required. SHOULD items are strongly recommended. DECIDE items are choices for
   the maintainers (Chris Carroll, Matt White). Do not settle a DECIDE item silently: propose an answer and say why.
3. **Follow HARK's process.** `docs/guides/contributing.md` (ll. 35-42) asks for an issue before a pull request for anything
   non-trivial. Open the issue from section 9 first.
4. **Keep the default path byte-identical.** Every existing call must give the same atoms, probabilities, `limit` contents and
   random streams. The stream-golden tests (`tests/test_distribution.py`, `StreamInvarianceGoldens`, ll. 1265-1292) must pass
   unchanged.

## 1. Why

- **Theory needs bounded shocks.** Buffer-stock theory works with shocks whose support is bounded away from zero and infinity.
  HARK already models this: every continuous distribution carries an `infimum` and a `supremum`. `discretize(...,
  endpoints=True)` adds zero-mass atoms at them, and `Distribution.discretize` copies them into the discrete distribution's
  `limit`. The purpose, in the words of `examples/Gentle-Intro/Advanced-Intro.ipynb` (ll. 624-632), is "so that HARK's solvers
  can use ... information about the 'best' and 'worst' things that can happen, even if they don't appear in the
  discretization." Otherwise the lowest possible realization is the lowest atom, which moves with the number of points.
- **The lognormal's bounds are useless here.** They are 0 and infinity. A truncated lognormal has finite bounds that do not
  depend on the number of points.
- **Research code needs the continuous object.** Solvers that integrate a shock against its density instead of summing over
  atoms (closed-form expectations, split quadrature at a kink) need the truncated distribution's density, cdf and partial
  moments. Their simulations then need equiprobable atoms of the same truncated distribution, so that the solver and the
  simulation hold the same beliefs. This need comes from HAFiscal's Step 1 estimation. There, integrating both income shocks
  continuously on a truncated lognormal (+-2 standard deviations) removes every slope kink in the consumption function except
  the borrowing constraint's (the truncation's edges leave curvature breaks). On a fixed grid, four far starting points then find
  one minimum.
- **HARK once had it.** HARK had a truncated lognormal in 2017: `approxLognormal` with `tail_N=0` and a `tail_bound`, by Matt
  White in commit `8a80a3fb`, "Truncated lognormal will be used to correctly handle the consumption floor in the 'robust' ind
  shock solver". The feature was lost in later refactors.

## 2. What exists now (verified at `d741d378`)

### 2.1 Distribution architecture (`HARK/distributions/`)

- **`Distribution`** (`base.py` ll. 95-237).
  - `__init__(self, seed=None)`; `None` draws an entropy seed (ll. 77-92). It sets placeholders `self.infimum = np.array([])` and
    `self.supremum = np.array([])` ("should be overwritten by subclasses", ll. 123-125).
  - `draw(N)` returns `self.rvs(size=N, random_state=self._rng).T` (l. 196), so a scipy-style `rvs` is required.
  - `discretize(self, N, method="equiprobable", endpoints=False, **kwds)` (ll. 198-237) dispatches to `"_approx_" + method`. It
    passes `endpoints` POSITIONALLY as the second argument, then copies `self.infimum` and `self.supremum` into
    `discretized_dstn.limit` (ll. 235-236).
- **`ContinuousFrozenDistribution(rv_continuous_frozen, Distribution)`** (`continuous.py` ll. 47-66) wraps a scipy
  `rv_continuous`; `__init__(self, dist, *args, seed=0, **kwds)`. It is not in `__all__`.
- **`Lognormal`** (`continuous.py` ll. 190-399; alias `LogNormal`, l. 435).
  - `__new__` and `__init__(mu=0.0, sigma=1.0, seed=None, mean=None, std=None)`; there is also a `from_mean_std` classmethod
    (ll. 401-432).
  - scipy backing: `lognorm, s=sigma, scale=exp(mu)` (l. 272). Bounds `np.array([0.0])` and `np.array([np.inf])` (ll. 273-274).
  - When `sigma == 0`, `__new__` returns a single-atom `DiscreteDistribution` (ll. 206-243).
  - `_approx_equiprobable(self, N, endpoints=False, tail_N=0, tail_bound=None, tail_order=np.e)` (ll. 276-381). Each atom is its
    bin's conditional mean, computed exactly through `erfc`, so the mean is preserved. `endpoints=True` appends zero-mass atoms at
    0 and infinity (ll. 362-364). `_approx_hermite` is at ll. 383-399.
- **`MeanOneLogNormal(sigma=1.0, seed=0)`** (ll. 438-445) sets `mu = -sigma**2/2`.
- **Arrays are not really supported.** Docstrings say `mu`/`sigma` may be "float or [float]", but numpy arrays fail at
  `if sigma == 0`, and lists fail in `discretize`. Time variation lives outside the class, in `IndexDistribution` (`base.py`) or
  per-period lists.
- **`__all__`** (`__init__.py` ll. 1-31) exports `Lognormal`, `LogNormal`, `MeanOneLogNormal`, `Normal`, `Uniform`, `Weibull`,
  the discrete and multivariate classes, and the utilities. The API reference (`docs/reference/tools/distribution.rst`) is an
  `automodule` that respects `__all__`.

### 2.2 Truncation in HARK today

- **No univariate truncation.** In `Lognormal._approx_equiprobable`, `tail_bound` means "CDF boundaries of the tails vs main
  portion ... Inoperative when tail_N = 0" (`continuous.py` ll. 291-294). It refines the tails with extra atoms over the FULL
  support; it does not truncate. It must be a list, and a float raises `TypeError`.
- **The multivariate truncation is partial** (`MultivariateLogNormal._approx_equiprobable(N, endpoints, tail_bound, decomp)`,
  `multivariate.py` ll. 397-487; PR #1412). It truncates in CDF terms ("By default the distribution is not truncated") and
  renormalizes; the discretized mean equals the analytic truncated mean. But it truncates in the factor space after
  Cholesky/eigen decomposition, not each marginal. It is silently ignored for a diagonal covariance, which is forwarded to the
  univariate method (ll. 284-301). The diagonal case with `endpoints=True` crashes. It never sets `infimum`/`supremum`. No test
  exercises `tail_bound`, and it has no CHANGELOG entry.
- **Normal-approximation helpers truncate at +-bound standard deviations and renormalize:** `make_markov_approx_to_normal(...,
  bound=3.5)` (`utils.py` ll. 65-131) and `make_tauchen_ar1(..., bound=3.0)` (ll. 184-230).

### 2.3 How the bounds travel, and where they break

- **Into `limit`:** `Distribution.discretize` copies the bounds (`base.py` ll. 235-236).
- **`add_discrete_outcome_constant_mean`** (`utils.py` ll. 289-328) rescales the atoms by `(1 - p x)/(1 - p)` (l. 325). It takes
  the bounds from the UNSCALED `limit` through `_bounds_with_added_atom` (ll. 238-264). This is harmless for (0, infinity) but
  wrong for finite bounds. `MixtureTranIncShk_HANK` rescales its atoms the same way and leaves `limit` alone
  (`IncomeProcesses.py` ll. 307-309).
- **`combine_indep_dstns`** (`utils.py` ll. 358-450) concatenates each component's `limit["infimum"/"supremum"]` and raises
  `KeyError` if one is missing. It asserts that the probabilities sum to one (l. 419).
- **`make_univariate`** (`discrete.py` ll. 576-613), used by `get_PermShkDstn_from_IncShkDstn` and its transitory twin, rebuilds a
  default `limit` from the atoms; the continuous bounds are lost.
- **The solvers ignore the bounds.** The readers are `calc_worst_inc_prob` and `calc_boro_const_nat`
  (`ConsumptionSaving/ConsIndShockModel.py` ll. 493-531), through `use_infimum`. Every call site passes False or the default
  False (ll. 688, 698-700, 934, 945-951; `ConsWealthUtilityModel.py` ll. 334-336). The default was turned off in #1589 ("did not
  work properly when vFunc=True"). Several solvers read the atoms directly (ConsBequest, ConsGenIncProcess, ConsMarkov,
  ConsIndShockModelFast, `calc_limiting_values`, `calc_bounding_values`).

### 2.4 The income-process layer (`HARK/Calibration/Income/IncomeProcesses.py`)

- **`LognormPermIncShk(sigma, n_approx, neutral_measure=False, seed=0)`** (ll. 168-211) discretizes
  `MeanOneLogNormal(sigma)` equiprobably with `tail_N=0` (positional sigma). It builds `limit` by hand. Under
  `neutral_measure` it multiplies the probabilities by the atoms without renormalizing (ll. 205-207), so it relies on an exact
  mean of one.
- **`MixtureTranIncShk(sigma, UnempPrb, IncUnemp, n_approx, seed=0)`** (ll. 214-255) uses the same discretization and then
  `add_discrete_outcome_constant_mean`. `MixtureTranIncShk_HANK` is at ll. 258-315.
- **`BufferStockIncShkDstn(sigma_Perm, sigma_Tran, n_approx_Perm, n_approx_Tran, UnempPrb, IncUnemp, neutral_measure=False,
  seed=0)`** (ll. 318-383) is a `DiscreteDistributionLabeled` with `var_names=["PermShk", "TranShk"]`. The order and the labels
  are load-bearing: solvers unpack `perm, tran = dstn.atoms` and call `expected(lambda x: x["PermShk"] * x["TranShk"], ...)`.
- **`construct_lognormal_income_process_unemployment(T_cycle, PermShkStd, PermShkCount, TranShkStd, TranShkCount, T_retire,
  UnempPrb, IncUnemp, UnempPrbRet, IncUnempRet, RNG, neutral_measure=False)`** (ll. 523-628) returns
  `IndexDistribution(engine=BufferStockIncShkDstn, conditional=_build_lifecycle_conditional(...), RNG=RNG, seed=...)`. The
  conditional keys are built at ll. 64-99 and 122-133. In retirement the standard deviations are 0 and the counts 1. It is the
  `"IncShkDstn"` constructor of about 16 agent types. Its siblings are the Markov, HANK and medical-expense constructors
  (ll. 731-857, 860-964, 967-1139).
- **How parameters reach constructors** (`core.py` ll. 707-776, `utilities.py` ll. 52-68).
  - Argument names are read with `get_arg_names` and resolved from the agent's attributes or parameters. A missing argument that
    has a default is skipped.
  - Defaults live in `IndShockConsumerType_IncShkDstn_default` (`ConsIndShockModel.py` ll. 2028-2038): `PermShkStd: [0.1]`,
    `PermShkCount: 7`, `TranShkStd: [0.1]`, `TranShkCount: 7`, and so on.
  - Time-varying parameters must be Python lists, because `IndexDistribution` dispatches on `type(item) is list` (`base.py`
    ll. 634-648), and engines must accept `seed=` as a keyword.

### 2.5 Tests, documentation, tooling

- **Tests.** Distribution tests are in `tests/test_distribution.py` (unittest classes run by `pytest -n auto`; tolerances
  `assertAlmostEqual` or `HARK_PRECISION = 4` from `tests/__init__.py`). Income-process tests are in
  `tests/Calibration/Income/test_IncomeProcesses.py`, which has none for `LognormPermIncShk` / `MixtureTranIncShk` /
  `BufferStockIncShkDstn`.
- **Documentation.** Docstrings are numpydoc (contributing.md ll. 214-221). Sphinx builds with `-W` in CI, so docstring warnings
  fail the build.
- **CHANGELOG.** `docs/CHANGELOG.md` has a `0.17.3 (dev)` section with Release Notes / Major / Minor Changes. Entries are one
  paragraph ending in a PR link.
- **Notebooks.** The ones in `examples/Distributions/` (including `EquiprobableLognormal.ipynb`) are executed in CI by nbval
  (`examples.yml`).
- **Tooling.** ruff `v0.11.8` with `ruff-format` via pre-commit; Python `>=3.12` (`pyproject.toml`); `scipy>=1.10`
  (`requirements/base.txt`). The PR template asks for tests, documentation and a CHANGELOG entry.

## 3. The mathematics (closed forms the implementation needs)

Z is standard normal, with cdf Phi and density phi. A truncated lognormal TLN(mu, sigma; a, b), with -inf <= a < b <= +inf in
STANDARD DEVIATIONS of the underlying normal, is

    X = exp(mu + sigma Z),   Z conditioned on a <= Z <= b,   M = Phi(b) - Phi(a)   (the retained mass).

With a = -inf and b = +inf it is `Lognormal(mu, sigma)` exactly. Write z(x) = (ln x - mu) / sigma.

| object | closed form |
|---|---|
| support (infimum, supremum) | [exp(mu + sigma a), exp(mu + sigma b)]  (0 and +inf for infinite bounds) |
| pdf | f(x) = phi(z(x)) / (sigma x M) for a <= z(x) <= b, else 0 |
| cdf | F(x) = [Phi(clip(z(x), a, b)) - Phi(a)] / M |
| ppf | F^-1(q) = exp(mu + sigma Phi^-1(Phi(a) + q M)) |
| raw moments | E[X^k] = exp(k mu + k^2 sigma^2 / 2) [Phi(b - k sigma) - Phi(a - k sigma)] / M  (any real k) |
| partial moments | E[X^k 1{lo < X < hi}] = exp(k mu + k^2 sigma^2 / 2) [Phi(min(z(hi), b) - k sigma) - Phi(max(z(lo), a) - k sigma)]^+ / M |
| mean-one location | mu = -sigma^2 / 2 - ln{ [Phi(b - sigma) - Phi(a - sigma)] / M }, so that E[X] = 1 after truncation |
| draws | X = F^-1(U), U uniform: inverse-cdf sampling with the distribution's seeded RNG |

**Equiprobable discretization with N points** (the analogue of `Lognormal._approx_equiprobable` with `tail_N=0`). Cut the
TRUNCATED cdf at q_j = j/N, j = 0..N, that is at the z-cuts c_j = Phi^-1(Phi(a) + q_j M) (c_0 = a, c_N = b). Each atom is its
bin's conditional mean and has probability 1/N:

    x_j = E[X | c_{j-1} < Z < c_j] = exp(mu + sigma^2 / 2) [Phi(c_j - sigma) - Phi(c_{j-1} - sigma)] / (M / N).

The atoms preserve the mean exactly. With a = -inf and b = +inf this is the formula HARK already evaluates through `erfc`.
Checked: for sigma = 0.3633 it reproduces `MeanOneLogNormal(0.3633).discretize(5, "equiprobable", tail_N=0)` to 1e-15.

**Precision.** Differences of Phi deep in the upper tail lose digits. Take them on the complementary side (Phi(-lo) - Phi(-hi))
or with `scipy.special.log_ndtr`, as `Lognormal._approx_equiprobable` does with its flagged low-tail branch.

**Reference numbers** (HAFiscal's Step 1 calibration, quarterly; truncation a = -2, b = +2, M = 0.9544997361; mean one):

| quantity | value |
|---|---|
| transitory sigma | 0.363318042491699 (sqrt 0.132) |
| mean-one mu, transitory | -0.0507940676 (untruncated: -0.066) |
| transitory support | [0.459586, 1.965687] |
| after HARK's unemployment rescaling (UnempPrb 0.044, IncUnemp 0.60, factor 1.0184100418) | [0.468047, 2.001876] |
| 5 equiprobable atoms, truncated | 0.609257, 0.792542, 0.951686, 1.143653, 1.502862 (mean 1) |
| 5 equiprobable atoms, untruncated (HARK today) | 0.570567, 0.773072, 0.937442, 1.137833, 1.581086 |
| permanent sigma | 0.03162277660168379 (sqrt 0.001) |
| mean-one mu, permanent | -0.000386854826 |
| permanent support | [0.938350, 1.064876] |

Note that the employed transitory infimum (0.468) lies below the unemployment income (0.60). The overall worst transitory
outcome is then an employed one.

## 4. Requirements

### 4.1 The distribution

- **R1 (MUST) A class for the truncated lognormal** in `HARK/distributions/continuous.py`, exported in `__all__` so that the API
  reference shows it, with an alias in the style of `LogNormal`. **DECIDE:** a separate class (`TruncatedLognormal`, recommended,
  which leaves `Lognormal` byte-identical) or a truncation keyword on `Lognormal`.
- **R2 (MUST) Parameters** `mu`, `sigma`, and the bounds in standard deviations of ln X (the use case is +-2). Infinite bounds
  give the untruncated lognormal. **DECIDE:**
  - the bound parameters' names and whether a single positive float means the symmetric interval [-b, b];
  - whether to accept bounds in CDF terms too.
  - Do NOT reuse the name `tail_bound` (section 2.2: it means two different things already).
- **R3 (MUST) A mean-one variant** (a class like `MeanOneLogNormal`, or a classmethod) that sets mu by the formula in section 3,
  so that E[X] = 1 AFTER the truncation. This is the truncate-and-renormalize semantics of the multivariate precedent and of
  HAFiscal's research code. **DECIDE:** confirm this against the alternative, the truncation of an untruncated mean-one lognormal
  (mean slightly below one).
- **R4 (MUST) scipy backing.** A small `rv_continuous` subclass with closed-form `_pdf`, `_logpdf`, `_cdf`, `_ppf` and raw moments
  (via `scipy.special.ndtr`, `ndtri`, `log_ndtr`), so that `ContinuousFrozenDistribution`, `rvs` and `draw` work unchanged. Do not
  use `scipy.stats.truncate`: it belongs to scipy's new random-variable infrastructure, not `rv_continuous`, and HARK supports
  `scipy>=1.10`.
- **R5 (MUST) Bounds.** `infimum = np.array([exp(mu + sigma a)])` and `supremum = np.array([exp(mu + sigma b)])`, always 1-D
  arrays (the documented contract), with 0 and inf for infinite bounds.
- **R6 (MUST) sigma = 0** returns a single-atom `DiscreteDistribution`, as `Lognormal.__new__` does: retirement periods pass
  sigma = 0. **SHOULD:** avoid `MeanOneLogNormal`'s quirk, where a positional sigma = 0 binds to `__new__`'s `mu` and does not
  return the single atom.
- **R7 (MUST) Seeds.** Accept `seed=` as a keyword, since `IndexDistribution` engines pass it. **DECIDE:** the default seed; the
  mean-one and income classes default to 0, the others to None.
- **R8 (MUST) Closed-form helpers** for solver-side use:
  - a vectorized `partial_moment(k, lo, hi)` returning E[X^k 1{lo < X < hi}] for real k and array `lo`, `hi`;
  - a `moment(k)` that accepts real k.
  - **SHOULD:** a method returning the standard-normal coordinate z(x) and the retained mass M, which quadrature in z needs.
- **R9 (MUST) Scalar parameters.** Time variation through `IndexDistribution` and lists, as for every other distribution.

### 4.2 Discretization

- **D1 (MUST)** `_approx_equiprobable(self, N, endpoints=False)`, with `endpoints` as the second positional argument: N equal-mass
  bins of the truncated distribution, atoms at the bins' conditional means (section 3), probabilities 1/N. The mean is exact,
  which `neutral_measure` and the sum-to-one assert of `combine_indep_dstns` need.
- **D2 (MUST)** `endpoints=True` appends zero-mass atoms at the finite infimum and supremum. `expected()` stays finite; with
  `Lognormal`'s infinite endpoints it returns nan.
- **D3 (MUST)** `limit` records `dist`, `method`, `N`, `endpoints` and the bounds; `infimum`/`supremum` arrive through the base
  `discretize`.
- **D4 (MUST)** Infinite bounds reproduce `Lognormal._approx_equiprobable(N, tail_N=0)` and `MeanOneLogNormal`'s discretization
  to 1e-14 relative.
- **D5 (SHOULD / DECIDE)** A Gauss-Legendre rule on [a, b] against phi/M, as `_approx_gauss_legendre`, for quadrature use.
  Gauss-Hermite is not natural on a truncated support.

### 4.3 Bounds propagation

- **P1 (MUST, test)** The truncation bounds reach `limit` through `discretize`, `add_discrete_outcome_constant_mean`,
  `combine_indep_dstns` and `BufferStockIncShkDstn`.
- **P2 (SHOULD; may be its own small PR)** `add_discrete_outcome_constant_mean` rescales the inherited bounds by the same factor
  as the atoms, and so does `MixtureTranIncShk_HANK`. Without this the transitory bounds in `limit` are off by
  (1 - p x)/(1 - p), which is 1.84 % in the reference calibration.
- **P3 (SHOULD)** `make_univariate` preserves the continuous bounds, or its docstring says it does not.
- **P4 (out of scope; document only)** Making solvers use the bounds (`use_infimum`) is a separate decision with history
  (#1589). Verify and document that `use_infimum=True` returns the truncated bounds. Note that WorstIncPrb is then 0, because the
  bounds carry no mass.

### 4.4 Income-process integration (default-preserving)

- **I1 (MUST)** A truncation argument, default None, on `LognormPermIncShk`, `MixtureTranIncShk` and (SHOULD)
  `MixtureTranIncShk_HANK`. When set, they discretize the mean-one truncated distribution instead of `MeanOneLogNormal`. When None,
  the atoms, probabilities, `limit` and random streams are byte-identical.
- **I2 (MUST)**
  - `BufferStockIncShkDstn` passes the permanent and transitory truncations through.
  - `construct_lognormal_income_process_unemployment` takes trailing keyword arguments with default None. Their names are the
    agent-parameter keys (see I3), because `get_arg_names` reads them.
  - `_build_lifecycle_conditional` carries them. In retirement, with sigma = 0, they have no effect.
- **I3 (MUST / DECIDE) The agent parameters.** Suggested names are `PermShkTrunc` and `TranShkTrunc`. contributing.md's
  abbreviation list (ll. 225-305) has `Shk`, `Std` and `Min`/`Max` but no word for truncation, so **DECIDE** the name.
  - Default None in `IndShockConsumerType_IncShkDstn_default`.
  - Recommended: time-invariant, like `PermShkCount`.
  - **DECIDE:** a float meaning a symmetric bound, a pair, or both; time-varying lists later if wanted.
- **I4 (SHOULD)** The Markov, HANK and medical-expense constructors accept the same parameters.
- **I5 (MUST)** `neutral_measure` with truncation works, which needs the exact mean of D1.

## 5. Tests (MUST; unittest style, in `tests/test_distribution.py` and `tests/Calibration/Income/test_IncomeProcesses.py`)

1. The pdf integrates to 1 over [infimum, supremum]; cdf and ppf are inverses; `rvs` sample mean and variance match the analytic
   values (seeded).
2. The mean-one class has E[X] = 1; raw moments and `partial_moment` match numerical quadrature for several k and intervals,
   including intervals that straddle the bounds.
3. Equiprobable discretization:
   - probabilities sum to one with equal masses;
   - the mean is exact;
   - atoms lie inside their bins and inside [infimum, supremum];
   - E[f] converges as N grows for a smooth f.
4. Infinite bounds reproduce `Lognormal` and `MeanOneLogNormal` discretizations to 1e-14 (D4).
5. `endpoints=True`: zero-mass atoms at the finite bounds, and `expected()` stays finite.
6. Bounds in `limit` after `add_discrete_outcome_constant_mean` (scaled, if P2 is done), `combine_indep_dstns` and
   `BufferStockIncShkDstn`; `calc_worst_inc_prob` and `calc_boro_const_nat` with `use_infimum=True` see them.
7. Income process:
   - truncation None is byte-identical (atoms, probabilities, `limit`, and the stream goldens);
   - with a truncation, the means are exact and the bounds right;
   - an `IndShockConsumerType` with the new parameters solves and simulates;
   - the engine works inside `IndexDistribution` with a `seed=` keyword.
8. sigma = 0 returns a single atom.
9. The reference numbers of section 3.

## 6. Documentation and housekeeping (MUST)

- numpydoc docstrings that build under Sphinx `-W`; the class in `__all__`.
- A CHANGELOG bullet under `0.17.3 (dev)` with the PR link. If it is opt-in: "Default None; the default call is unchanged."
- A short section in `examples/Distributions/EquiprobableLognormal.ipynb` or a new small notebook. Notebooks run in CI, so keep it
  fast.
- ruff clean; Python `>=3.12`; an issue before the pull request.

## 7. Related defects found while surveying (separate issues, not part of this feature)

| item | where | what |
|---|---|---|
| Uniform bounds | `continuous.py` ll. 471-472 | `infimum`/`supremum` are `[0, inf]` whatever `bot`/`top` are |
| Weibull | `continuous.py` | no `_approx_*` method, so `discretize` raises |
| multivariate `tail_bound` | `multivariate.py` | ignored for diagonal covariance; diagonal + `endpoints=True` crashes; bounds never set; untested |
| bounds after rescaling | `utils.py` l. 325; `IncomeProcesses.py` ll. 307-309 | atoms rescaled, `limit` bounds not (P2) |
| `make_univariate` | `discrete.py` ll. 576-613 | drops the continuous bounds (P3) |
| `MeanOneLogNormal(0.0)` | `continuous.py` | positional sigma = 0 binds to `mu` and does not return the single atom |
| seed docstrings | `multivariate.py` ll. 107-115; `discrete.py` ll. 111, 121; `utils.py` ll. 358-369 | say "default 0", code has None |
| `IndexDistribution(RNG=...)` | `base.py` ll. 603-609 | the passed RNG is replaced by a seeded one, contrary to the docstring |
| HANK constructor | `IncomeProcesses.py` ll. 860-964 | passes no seed: not reproducible |
| contributing.md | ll. 117-118, 178, 196, 211 | `master` (the branch is `main`), Python 3.10-3.13 (now `>=3.12`), `pytest HARK/` (tests are in `tests/`) |

## 8. Out of scope

- Changing any default.
- Making solvers use the bounds.
- Unifying the univariate and multivariate truncation conventions; this feature should not foreclose that.
- Any HAFiscal-side adoption.

## 9. Suggested issue text (for econ-ark/HARK, with the maintainers' permission)

> **Feature: a truncated continuous lognormal (and truncated income shocks).** HARK's distributions carry `infimum`/`supremum` so
> that solvers can know the worst and best outcomes independently of the discretization, but the lognormal's bounds are 0 and
> infinity. A truncated lognormal (bounds in standard deviations of ln X, mean-one variant, closed-form pdf/cdf/ppf, partial
> moments, equiprobable discretization with exact mean, finite endpoints) would give finite, N-independent bounds and would serve
> research code that integrates shocks against their densities. Opt-in `PermShkTrunc`/`TranShkTrunc` on the lognormal income
> constructors, default None (byte-identical). HARK had this in 2017 (`approxLognormal` with `tail_N=0`, `tail_bound`; 8a80a3fb)
> and lost it. Full requirements: `.agents/feature-specs/truncated-lognormal-requirements.md`.
