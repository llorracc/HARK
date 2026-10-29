from typing import Any, Optional, Union

import numpy as np
from scipy.special import erfc, ndtr, ndtri
from scipy.stats import (
    rv_continuous,
    norm,
    lognorm,
    uniform,
    weibull_min,
)
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from HARK.distributions.base import Distribution, _renamed_class
from HARK.distributions.discrete import DiscreteDistribution


def _hermite_discretize(distribution, N, endpoints, atoms_transform, lower_endpoint):
    """Gauss-Hermite quadrature discretization shared by Normal/LogNormal.

    ``atoms_transform`` maps the linear ``sqrt(2)*sigma*x + mu`` grid to the
    distribution's support (identity for Normal, ``np.exp`` for LogNormal).
    ``lower_endpoint`` is the value prepended when ``endpoints=True``.
    """
    x, w = np.polynomial.hermite.hermgauss(N)
    pmv = w * np.pi**-0.5
    atoms = atoms_transform(np.sqrt(2.0) * distribution.sigma * x + distribution.mu)
    if endpoints:
        atoms = np.r_[lower_endpoint, atoms, np.inf]
        pmv = np.r_[0.0, pmv, 0.0]
    return DiscreteDistribution(
        pmv,
        atoms,
        seed=distribution.random_seed(),
        limit={
            "dist": distribution,
            "method": "hermite",
            "N": N,
            "endpoints": endpoints,
        },
    )


def _log_coordinate(mu, sigma):
    """The standard normal coordinate (ln x - mu) / sigma, -inf where x <= 0."""

    def coord(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                x > 0.0, (np.log(np.where(x > 0.0, x, 1.0)) - mu) / sigma, -np.inf
            )

    return coord


# CONTINUOUS DISTRIBUTIONS


class ContinuousFrozenDistribution(rv_continuous_frozen, Distribution):
    """
    Parameterized continuous distribution from scipy.stats with seed management.
    """

    def __init__(
        self, dist: rv_continuous, *args: Any, seed: int = 0, **kwds: Any
    ) -> None:
        """
        Parameterized continuous distribution from scipy.stats with seed management.

        Parameters
        ----------
        dist : rv_continuous
            Continuous distribution from scipy.stats.
        seed : int, optional
            Seed for random number generator, by default 0
        """
        rv_continuous_frozen.__init__(self, dist, *args, **kwds)
        Distribution.__init__(self, seed=seed)


class Normal(ContinuousFrozenDistribution):
    """
    A Normal distribution.

    Parameters
    ----------
    mu : float or [float]
        One or more means.  Number of elements T in mu determines number
        of rows of output.
    sigma : float or [float]
        One or more standard deviations. Number of elements T in sigma
        determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    def __new__(
        cls,
        mu: Union[float, np.ndarray] = 0.0,
        sigma: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Create a new Normal distribution. If sigma is zero, return a
        DiscreteDistribution with a single atom at mu.
        Parameters
        ----------
        mu : Union[float, np.ndarray], optional
            Mean of normal distribution, by default 0.0
        sigma : Union[float, np.ndarray], optional
            Standard deviation of normal distribution, by default 1.0
        seed : Optional[int], optional
            Seed for random number generator, by default None
        Returns
        -------
        Normal or DiscreteDistribution
            Normal distribution or DiscreteDistribution with a single atom.
        """

        if sigma == 0:
            # If sigma is zero, return a DiscreteDistribution with a single atom
            return DiscreteDistribution([1.0], mu, seed=seed)

        return super().__new__(cls)

    def __init__(self, mu=0.0, sigma=1.0, seed=None):
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)

        if self.mu.size != self.sigma.size:
            raise AttributeError(
                f"'mu' and 'sigma' must be of the same size. Instead 'mu' is of "
                f"size {self.mu.size} and 'sigma' is of size {self.sigma.size}."
            )

        super().__init__(norm, loc=mu, scale=sigma, seed=seed)
        self.infimum = -np.inf * np.ones(self.mu.size)
        self.supremum = np.inf * np.ones(self.mu.size)

    def discretize(self, N, method="hermite", endpoints=False):
        """
        For normal distributions, the Gauss-Hermite quadrature rule is
        used as the default method for discretization.
        """

        return super().discretize(N, method, endpoints)

    def normal_coordinate(self):
        """
        The coordinate in which this law is the standard normal, as
        expected_continuous() integrates in it: z = (x - mu) / sigma.

        Returns
        -------
        bot, top : float
            The bounds of z, here -inf and inf.
        value : callable
            The law's value at z.
        coord : callable
            The coordinate z of a value.
        """
        mu, sigma = float(np.squeeze(self.mu)), float(np.squeeze(self.sigma))
        return -np.inf, np.inf, lambda z: mu + sigma * z, lambda x: (x - mu) / sigma

    def _approx_hermite(self, N, endpoints=False):
        """
        Returns a discrete approximation of this distribution
        using the Gauss-Hermite quadrature rule.

        Parameters
        ----------
        N : int
            Number of discrete points to approximate the distribution.

        Returns
        -------
        DiscreteDistribution
            Discrete approximation of this distribution.
        """
        return _hermite_discretize(self, N, endpoints, lambda v: v, -np.inf)

    def _approx_equiprobable(self, N, endpoints=False):
        """
        Returns a discrete equiprobable approximation of this distribution.

        Parameters
        ----------
        N : int
            Number of discrete points to approximate the distribution.

        Returns
        -------
        DiscreteDistribution
            Discrete approximation of this distribution.
        """

        CDF = np.linspace(0, 1, N + 1)
        lims = norm.ppf(CDF)
        pdf = norm.pdf(lims)

        # Find conditional means using Mills's ratio
        pmv = np.diff(CDF)
        atoms = self.mu - np.diff(pdf) / pmv * self.sigma

        if endpoints:
            atoms = np.r_[-np.inf, atoms, np.inf]
            pmv = np.r_[0.0, pmv, 0.0]

        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self._rng.integers(0, 2**31 - 1, dtype="int32"),
            limit=limit,
        )


class LogNormal(ContinuousFrozenDistribution):
    """
    A LogNormal distribution

    Parameters
    ----------
    mu : float or [float]
        One or more means of underlying normal distribution.
        Number of elements T in mu determines number of rows of output.
    sigma : float or [float]
        One or more standard deviations of underlying normal distribution.
        Number of elements T in sigma determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    def __new__(
        cls,
        mu: Union[float, np.ndarray] = 0.0,
        sigma: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = None,
        mean=None,
        std=None,
    ):
        """
        Create a new LogNormal distribution. If sigma is zero, return a
        DiscreteDistribution with a single atom at exp(mu).

        Parameters
        ----------
        mu : Union[float, np.ndarray], optional
            Mean of underlying normal distribution, by default 0.0
        sigma : Union[float, np.ndarray], optional
            Standard deviation of underlying normal distribution, by default 1.0
        seed : Optional[int], optional
            Seed for random number generator, by default None
        mean: optional
            For alternative mean/std parameterization, the mean of the lognormal distribution
        std: optional
            For alternative mean/std parameterization, the standard deviation of the lognormal distribution

        Returns
        -------
        LogNormal or DiscreteDistribution
            LogNormal distribution or DiscreteDistribution with a single atom.
        """

        if sigma == 0:
            # If sigma is zero, return a DiscreteDistribution with a single atom
            if mean is not None:
                return DiscreteDistribution([1.0], mean, seed=seed)
            return DiscreteDistribution([1.0], [np.exp(mu)], seed=seed)

        return super().__new__(cls)

    def __init__(
        self,
        mu: Union[float, np.ndarray] = 0.0,
        sigma: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = None,
        mean=None,
        std=None,
    ):
        if mean is not None and std is not None:
            mean_squared = mean**2
            variance = std**2
            mu = np.log(mean / (np.sqrt(1.0 + variance / mean_squared)))
            sigma = np.sqrt(np.log(1.0 + variance / mean_squared))

        if mean is not None and std is None:
            mu = np.log(mean) - sigma**2 / 2

        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)

        if self.mu.size != self.sigma.size:
            raise AttributeError(
                "mu and sigma must be of same size, are %s, %s"
                % ((self.mu.size), (self.sigma.size))
            )

        # Set up the RNG
        super().__init__(lognorm, s=self.sigma, scale=np.exp(self.mu), loc=0, seed=seed)
        if np.all(self.sigma == 0.0):
            # Degenerate at exp(mu), as in retirement's permanent shock
            self.infimum = np.atleast_1d(np.exp(self.mu)).astype(float)
            self.supremum = self.infimum.copy()
        else:
            self.infimum = np.array([0.0])
            self.supremum = np.array([np.inf])

    def _approx_equiprobable(
        self, N, endpoints=False, tail_N=0, tail_bound=None, tail_order=np.e
    ):
        """
        Construct a discrete approximation to a lognormal distribution with underlying
        normal distribution N(mu,sigma).  Makes an equiprobable distribution by
        default, but user can optionally request augmented tails with exponentially
        sized point masses.  This can improve solution accuracy in some models.

        Parameters
        ----------
        N: int
            Number of discrete points in the "main part" of the approximation.
        tail_N: int
            Number of points in each "tail part" of the approximation; 0 = no tail.
        tail_bound: [float]
            CDF boundaries of the tails vs main portion; tail_bound[0] is the lower
            tail bound, tail_bound[1] is the upper tail bound.  Inoperative when
            tail_N = 0.  Can make "one tailed" approximations with 0.0 or 1.0.
        tail_order: float
            Factor by which consecutive point masses in a "tail part" differ in
            probability.  Should be >= 1 for sensible spacing.

        Returns
        -------
        d : DiscreteDistribution
            Probability associated with each point in array of discrete
            points for discrete probability mass function.
        """
        tail_bound = tail_bound or [0.02, 0.98]

        # Handle the trivial case first
        if self.sigma == 0.0:
            pmv = np.ones(N) / N
            atoms = np.exp(self.mu) * np.ones(N)

        else:
            # Find the CDF boundaries of each segment
            if tail_N > 0:
                lo_cut = tail_bound[0]
                hi_cut = tail_bound[1]
            else:
                lo_cut = 0.0
                hi_cut = 1.0
            inner_size = hi_cut - lo_cut
            inner_CDF_vals = [
                lo_cut + x * N ** (-1.0) * inner_size for x in range(1, N)
            ]
            if inner_size < 1.0:
                scale = 1.0 / tail_order
                mag = (1.0 - scale**tail_N) / (1.0 - scale)
            lower_CDF_vals = [0.0]
            if lo_cut > 0.0:
                for x in range(tail_N - 1, -1, -1):
                    lower_CDF_vals.append(lower_CDF_vals[-1] + lo_cut * scale**x / mag)
            upper_CDF_vals = [hi_cut]
            if hi_cut < 1.0:
                for x in range(tail_N):
                    upper_CDF_vals.append(
                        upper_CDF_vals[-1] + (1.0 - hi_cut) * scale**x / mag
                    )
            CDF_vals = np.array(lower_CDF_vals + inner_CDF_vals + upper_CDF_vals)
            CDF_vals[-1] = 1.0
            CDF_vals[0] = 0.0  # somehow these need fixing sometimes

            # Calculate probability masses for each node
            pmv = CDF_vals[1:] - CDF_vals[:-1]
            pmv /= np.sum(pmv)

            # Translate the CDF values to z-scores (stdevs from mean), then get q-scores
            z_cuts = norm.ppf(CDF_vals)
            q_cuts = (z_cuts - self.sigma) / np.sqrt(2)

            # Evaluate the (complementary) error function at the q values
            erf_q = erfc(q_cuts)
            erf_q_neg = erfc(-q_cuts)

            # Evaluate the base for the conditional expectations
            vals_base = erf_q[:-1] - erf_q[1:]
            these = q_cuts[:-1] < -2.0  # flag low q values and use the *other* version
            vals_base[these] = erf_q_neg[1:][these] - erf_q_neg[:-1][these]

            # Make and apply the normalization factor and probability weights
            norm_fac = 0.5 * np.exp(self.mu + 0.5 * self.sigma**2) / pmv
            atoms = vals_base * norm_fac

        if endpoints:
            atoms = np.r_[0.0, atoms, np.inf]
            pmv = np.r_[0.0, pmv, 0.0]

        limit = {
            "dist": self,
            "method": "equiprobable",
            "N": N,
            "endpoints": endpoints,
            "tail_N": tail_N,
            "tail_bound": tail_bound,
            "tail_order": tail_order,
        }

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self.random_seed(),
            limit=limit,
        )

    def _approx_hermite(self, N, endpoints=False):
        """
        Returns a discrete approximation of this distribution
        using the Gauss-Hermite quadrature rule.

        Parameters
        ----------
        N : int
            Number of discrete points to approximate the distribution.

        Returns
        -------
        DiscreteDistribution
            Discrete approximation of this distribution.
        """

        return _hermite_discretize(self, N, endpoints, np.exp, 0.0)

    def normal_coordinate(self):
        """
        The coordinate in which this law is the standard normal, as
        expected_continuous() integrates in it: z = (ln x - mu) / sigma.

        Returns
        -------
        bot, top : float
            The bounds of z, here -inf and inf.
        value : callable
            The law's value at z.
        coord : callable
            The coordinate z of a value, -inf at values not above zero.
        """
        mu, sigma = float(np.squeeze(self.mu)), float(np.squeeze(self.sigma))
        return (
            -np.inf,
            np.inf,
            lambda z: np.exp(mu + sigma * z),
            _log_coordinate(mu, sigma),
        )

    @classmethod
    def from_mean_std(cls, mean, std, seed=None):
        """
        Construct a LogNormal distribution from its
        mean and standard deviation.

        This is unlike the normal constructor for the distribution,
        which takes the mu and sigma for the normal distribution
        that is the logarithm of the Log Normal distribution.

        Parameters
        ----------
        mean : float or [float]
            One or more means.  Number of elements T in mu determines number
            of rows of output.
        std : float or [float]
            One or more standard deviations. Number of elements T in sigma
            determines number of rows of output.
        seed : int
            Seed for random number generator.

        Returns
        ---------
        LogNormal

        """
        mean_squared = mean**2
        variance = std**2
        mu = np.log(mean / (np.sqrt(1.0 + variance / mean_squared)))
        sigma = np.sqrt(np.log(1.0 + variance / mean_squared))

        return cls(mu=mu, sigma=sigma, seed=seed)


class LogNormalMeanOne(LogNormal):
    """
    A LogNormal distribution with mean 1.
    """

    def __init__(self, sigma=1.0, seed=0):
        mu = -0.5 * sigma**2
        super().__init__(mu=mu, sigma=sigma, seed=seed)


def _normal_mass(lo, hi):
    """
    P(lo < Z < hi) for a standard normal Z, taken on the side of zero that
    keeps its digits in the tails.
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return np.where(lo > 0.0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))


def _truncated_normal_ppf(q, lo, hi):
    """
    Quantile q of a standard normal truncated to [lo, hi].
    """
    mass = _normal_mass(lo, hi)
    with np.errstate(invalid="ignore"):
        upper = -ndtri(ndtr(-lo) - q * mass)
        lower = ndtri(ndtr(lo) + q * mass)
    return np.where(np.asarray(lo) > 0.0, upper, lower)


def _bound(bound, default):
    """A truncation bound in standard deviations; None is no bound on that side."""
    return default if bound is None else float(bound)


def truncated_normal_variance(bot=-3.0, top=3.0):
    """
    The variance of a standard normal truncated to [bot, top], in standard
    deviations; None is no bound on that side. A lognormal truncated in the log
    at these bounds has Var(ln X) = sigma**2 times this factor.

    Parameters
    ----------
    bot : float or None, optional
        Lower bound, by default -3.0.
    top : float or None, optional
        Upper bound, by default 3.0.

    Returns
    -------
    float
    """
    lo, hi = _bound(bot, -np.inf), _bound(top, np.inf)
    mass = _normal_mass(lo, hi)
    pdf_lo, pdf_hi = norm.pdf(lo), norm.pdf(hi)
    z_pdf_lo = lo * pdf_lo if np.isfinite(lo) else 0.0
    z_pdf_hi = hi * pdf_hi if np.isfinite(hi) else 0.0
    return float(1.0 + (z_pdf_lo - z_pdf_hi) / mass - ((pdf_lo - pdf_hi) / mass) ** 2)


class _truncated_lognorm_gen(rv_continuous):
    """
    The lognormal exp(s Z), with Z a standard normal truncated to [lo, hi] in its
    standard deviations. scipy's scale is exp(mu).
    """

    def _argcheck(self, s, lo, hi):
        return (s > 0) & (lo < hi)

    def _get_support(self, s, lo, hi):
        return np.exp(s * lo), np.exp(s * hi)

    def _pdf(self, x, s, lo, hi):
        with np.errstate(divide="ignore"):
            z = np.log(x) / s
        inside = (z >= lo) & (z <= hi)
        dens = norm.pdf(z) / (s * x * _normal_mass(lo, hi))
        return np.where(inside, dens, 0.0)

    def _cdf(self, x, s, lo, hi):
        with np.errstate(divide="ignore"):
            z = np.clip(np.log(x) / s, lo, hi)
        return _normal_mass(lo, z) / _normal_mass(lo, hi)

    def _ppf(self, q, s, lo, hi):
        return np.exp(s * _truncated_normal_ppf(q, lo, hi))

    def _munp(self, n, s, lo, hi):
        return (
            np.exp(0.5 * (n * s) ** 2)
            * _normal_mass(lo - n * s, hi - n * s)
            / _normal_mass(lo, hi)
        )


_truncated_lognorm = _truncated_lognorm_gen(
    a=0.0, name="truncated_lognorm", shapes="s, lo, hi"
)


class LogNormalTruncated(ContinuousFrozenDistribution):
    """
    A lognormal distribution truncated in the log: X = exp(mu + sigma Z), with Z a
    standard normal conditioned on bot <= Z <= top. The bounds are in standard
    deviations of that normal, and None is no bound on that side, so with both
    None this is LogNormal(mu, sigma).

    Parameters
    ----------
    mu : float, optional
        Mean of the underlying normal before truncation, by default 0.0.
    sigma : float, optional
        Standard deviation of the underlying normal before truncation, by
        default 1.0. With sigma = 0 the distribution is the point exp(mu).
    bot : float or None, optional
        Lower truncation bound, by default -3.0.
    top : float or None, optional
        Upper truncation bound, by default 3.0.
    seed : int, optional
        Seed for random number generator, by default None.
    """

    def __new__(cls, mu=0.0, sigma=1.0, bot=-3.0, top=3.0, seed=None):
        if sigma == 0:
            # A point at exp(mu), whose bounds are the point
            return DiscreteDistribution([1.0], [np.exp(mu)], seed=seed)
        return super().__new__(cls)

    def __init__(self, mu=0.0, sigma=1.0, bot=-3.0, top=3.0, seed=None):
        if np.ndim(mu) or np.ndim(sigma):
            raise ValueError(
                "mu and sigma must be scalars; use IndexDistribution for time variation."
            )
        if sigma < 0:
            raise ValueError("sigma must be nonnegative, but it is " + str(sigma))
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.bot = _bound(bot, -np.inf)
        self.top = _bound(top, np.inf)
        if not self.bot < self.top:
            raise ValueError("The truncation bounds must satisfy bot < top.")
        super().__init__(
            _truncated_lognorm,
            s=self.sigma,
            lo=self.bot,
            hi=self.top,
            scale=np.exp(self.mu),
            loc=0,
            seed=seed,
        )
        self.infimum = np.array([np.exp(self.mu + self.sigma * self.bot)])
        self.supremum = np.array([np.exp(self.mu + self.sigma * self.top)])

    @property
    def mass(self):
        """The probability mass of the underlying normal that the truncation keeps."""
        return float(_normal_mass(self.bot, self.top))

    def z(self, x):
        """The standard normal coordinate (ln x - mu) / sigma of x."""
        with np.errstate(divide="ignore"):
            return (np.log(x) - self.mu) / self.sigma

    def normal_coordinate(self):
        """
        The coordinate in which this law is the standard normal truncated to
        [bot, top], as expected_continuous() integrates in it: the underlying
        normal's z = (ln x - mu) / sigma.

        Returns
        -------
        bot, top : float
            The bounds of z, the truncation's.
        value : callable
            The law's value at z.
        coord : callable
            The coordinate z of a value, -inf at values not above zero.
        """
        mu, sigma = self.mu, self.sigma
        return (
            self.bot,
            self.top,
            lambda z: np.exp(mu + sigma * z),
            _log_coordinate(mu, sigma),
        )

    def moment(self, order):
        """
        The raw moment E[X**order], for any real order, in closed form.
        """
        k = np.asarray(order, dtype=float)
        ks = k * self.sigma
        return (
            np.exp(k * self.mu + 0.5 * ks**2)
            * _normal_mass(self.bot - ks, self.top - ks)
            / self.mass
        )

    def partial_moment(self, order, lo, hi):
        """
        The partial moment E[X**order 1{lo < X < hi}], for any real order and
        arrays of interval ends.
        """
        k = np.asarray(order, dtype=float)
        ks = k * self.sigma
        z_lo = np.maximum(
            self.z(np.maximum(np.asarray(lo, dtype=float), 0.0)), self.bot
        )
        z_hi = np.minimum(self.z(np.asarray(hi, dtype=float)), self.top)
        mass = np.maximum(_normal_mass(z_lo - ks, z_hi - ks), 0.0)
        return np.where(
            z_hi > z_lo, np.exp(k * self.mu + 0.5 * ks**2) * mass / self.mass, 0.0
        )

    def _approx_equiprobable(self, N, endpoints=False):
        """
        A discrete approximation with N atoms of probability 1/N, each the
        conditional mean of its bin of the truncated distribution, so the mean is
        exact.

        Parameters
        ----------
        N : int
            Number of atoms.
        endpoints : bool, optional
            Whether to add zero-probability atoms at the infimum and supremum,
            by default False.

        Returns
        -------
        DiscreteDistribution
        """
        cuts = _truncated_normal_ppf(np.arange(N + 1) / N, self.bot, self.top)
        cuts[0], cuts[-1] = self.bot, self.top
        pmv = np.ones(N) / N
        bin_mass = _normal_mass(cuts[:-1] - self.sigma, cuts[1:] - self.sigma)
        atoms = np.exp(self.mu + 0.5 * self.sigma**2) * bin_mass / (self.mass / N)
        if endpoints:
            atoms = np.r_[self.infimum, atoms, self.supremum]
            pmv = np.r_[0.0, pmv, 0.0]
        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}
        return DiscreteDistribution(pmv, atoms, seed=self.random_seed(), limit=limit)


class LogNormalTruncatedMeanOne(LogNormalTruncated):
    """
    A truncated lognormal with mean one after the truncation: mu is set so that
    E[X] = 1 for the truncated distribution. The bounds are in standard
    deviations of the underlying normal, by default three either side.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the underlying normal before truncation, by
        default 1.0. With sigma = 0 the distribution is the point 1.
    bot : float or None, optional
        Lower truncation bound, by default -3.0.
    top : float or None, optional
        Upper truncation bound, by default 3.0.
    seed : int, optional
        Seed for random number generator, by default 0.
    """

    def __new__(cls, sigma=1.0, bot=-3.0, top=3.0, seed=0):
        if sigma == 0:
            return DiscreteDistribution([1.0], [1.0], seed=seed)
        return super().__new__(cls)

    def __init__(self, sigma=1.0, bot=-3.0, top=3.0, seed=0):
        lo, hi = _bound(bot, -np.inf), _bound(top, np.inf)
        mu = -0.5 * sigma**2 - np.log(
            _normal_mass(lo - sigma, hi - sigma) / _normal_mass(lo, hi)
        )
        super().__init__(mu=mu, sigma=sigma, bot=bot, top=top, seed=seed)

    @classmethod
    def from_sd_log(cls, sd_log, bot=-3.0, top=3.0, seed=0):
        """
        The mean-one truncated lognormal whose log has standard deviation
        sd_log after the truncation. The underlying normal's sigma is
        sd_log / sqrt(truncated_normal_variance(bot, top)), so an estimated
        dispersion keeps its value under the truncation.

        Parameters
        ----------
        sd_log : float
            Standard deviation of ln X after truncation.
        bot : float or None, optional
            Lower truncation bound, by default -3.0.
        top : float or None, optional
            Upper truncation bound, by default 3.0.
        seed : int, optional
            Seed for random number generator, by default 0.

        Returns
        -------
        LogNormalTruncatedMeanOne or DiscreteDistribution
        """
        sigma = sd_log / np.sqrt(truncated_normal_variance(bot, top))
        return cls(sigma=sigma, bot=bot, top=top, seed=seed)


def _weighted_quantile(y, w, q):
    """Quantile q of the points y with probabilities w, interpolated linearly."""
    order = np.argsort(y)
    y, w = y[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    return np.interp(q, cum / cum[-1], y)


class LogNormalMixture(Distribution):
    """
    A mixture of lognormal distributions, optionally truncated: with probability
    proportional to weights[k], X is exp(mu[k] + sigma[k] Z) for a standard
    normal Z, and with bounds the mixture as a whole is restricted to
    [lower, upper] and rescaled to unit mass.

    expected_continuous() integrates it component by component, through the
    attributes ``weights`` and ``components``: the components are the
    lognormals truncated to [lower, upper], each a LogNormalTruncated, and the
    weights are those of the truncated mixture. LogNormalMixture.from_points()
    builds one from a set of points by kernel smoothing in their logs, so that a
    distribution given by points, such as data, can enter a model as a
    continuous law, without mass points.

    It discretizes two ways: discretize(N), the default, splits the whole
    mixture into N atoms of equal probability, and
    discretize(N, method="components") splits each component into N, each
    carrying its component's weight over N.

    Parameters
    ----------
    mu : float or np.array
        Means of the components' underlying normals.
    sigma : float or np.array
        Standard deviations of the components' underlying normals, positive.
    weights : np.array, optional
        Probabilities of the components before any truncation, normalized to
        sum to one; equal by default.
    lower : float or None, optional
        Lower end of the support, a value of X; None, the default, is zero.
    upper : float or None, optional
        Upper end of the support, a value of X; None, the default, is no bound.
    seed : int, optional
        Seed for random number generator, by default None.
    """

    def __init__(self, mu, sigma, weights=None, lower=None, upper=None, seed=None):
        mu, sigma = np.broadcast_arrays(
            np.atleast_1d(np.asarray(mu, dtype=float)),
            np.atleast_1d(np.asarray(sigma, dtype=float)),
        )
        if mu.ndim != 1:
            raise ValueError("mu and sigma must be scalars or 1D arrays.")
        if not np.all(sigma > 0.0):
            raise ValueError("Every component's sigma must be positive.")
        if weights is None:
            weights = np.ones(mu.size)
        weights = np.asarray(weights, dtype=float)
        if weights.shape != mu.shape or np.any(weights < 0.0) or weights.sum() <= 0:
            raise ValueError(
                "weights must be nonnegative, one per component, and not all zero."
            )
        self.lower = 0.0 if lower is None else float(lower)
        self.upper = np.inf if upper is None else float(upper)
        if not 0.0 <= self.lower < self.upper:
            raise ValueError("The support must satisfy 0 <= lower < upper.")
        super().__init__(seed=seed)
        self.mu = mu.copy()
        self.sigma = sigma.copy()

        # Each component's bounds in its own standard-normal coordinate
        with np.errstate(divide="ignore"):
            self._bot = (np.log(self.lower) - self.mu) / self.sigma
            self._top = (np.log(self.upper) - self.mu) / self.sigma
        self._mass = _normal_mass(self._bot, self._top)
        with np.errstate(divide="ignore"):
            self._inv_mass = np.where(self._mass > 0.0, 1.0 / self._mass, 0.0)
        kept = weights * self._mass
        if not kept.sum() > 0.0:
            raise ValueError("No component has mass inside [lower, upper].")
        self.mass = float(kept.sum() / weights.sum())  # the share the truncation keeps
        self.weights = kept / kept.sum()
        self.components = [
            LogNormalTruncated(
                m,
                s,
                bot=None if np.isneginf(b) else b,
                top=None if np.isposinf(t) else t,
            )
            if p > 0.0
            else DiscreteDistribution([1.0], [np.exp(m)])
            for m, s, b, t, p in zip(self.mu, self.sigma, self._bot, self._top, kept)
        ]
        self.infimum = np.array([self.lower])
        self.supremum = np.array([self.upper])

    @classmethod
    def from_points(
        cls,
        points,
        weights=None,
        bandwidth=None,
        match_variance=True,
        seed=None,
    ):
        """
        A lognormal mixture from points by kernel smoothing in their logs: each
        point becomes a lognormal component whose underlying normal has the
        bandwidth as its standard deviation, and the mixture is truncated once,
        at three bandwidths beyond the outermost points.

        With match_variance (the default), the logs are first moved toward their
        mean by sqrt(1 - bandwidth**2 / variance), so that the smoothing gives
        back the variance of the points' logs (before the truncation, which
        trims a little); without it, the variance of the logs grows by the
        bandwidth squared, as in plain kernel smoothing. The mixture is then
        rescaled to the points' mean.

        Parameters
        ----------
        points : np.array
            Positive points, such as observed income shocks.
        weights : np.array, optional
            Probabilities of the points; equal by default.
        bandwidth : float, optional
            Standard deviation of each component's underlying normal. By default,
            Silverman's rule of thumb on the logs: 0.9 min(s, IQR / 1.34)
            n**(-1/5), with s and IQR the logs' standard deviation and
            interquartile range and n the points' effective number,
            1 / sum(weights**2).
        match_variance : bool, optional
            Whether the smoothing keeps the variance of the points' logs, by
            default True.
        seed : int, optional
            Seed for random number generator, by default None.

        Returns
        -------
        LogNormalMixture
        """
        x = np.asarray(points, dtype=float).ravel()
        if not np.all(x > 0.0):
            raise ValueError("The points must be positive.")
        w = np.ones(x.size) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != x.shape or np.any(w < 0.0) or w.sum() <= 0:
            raise ValueError("weights must be nonnegative, one per point.")
        w = w / w.sum()
        y = np.log(x)
        m = np.dot(w, y)
        var = np.dot(w, (y - m) ** 2)
        if not var > 0.0:
            raise ValueError("The points must take at least two values.")
        if bandwidth is None:
            iqr = _weighted_quantile(y, w, 0.75) - _weighted_quantile(y, w, 0.25)
            spread = min(np.sqrt(var), iqr / 1.34) if iqr > 0.0 else np.sqrt(var)
            bandwidth = 0.9 * spread * np.sum(w**2) ** 0.2
        h = float(bandwidth)
        if not h > 0.0:
            raise ValueError("bandwidth must be positive.")
        if match_variance:
            if h * h >= var:
                raise ValueError(
                    "To keep the variance of the logs, the bandwidth must be "
                    "smaller than their standard deviation."
                )
            y = m + np.sqrt(1.0 - h * h / var) * (y - m)
        mu = y - 0.5 * h * h
        lower, upper = np.exp(y.min() - 3.0 * h), np.exp(y.max() + 3.0 * h)
        unscaled = cls(mu, h, w, lower=lower, upper=upper)
        scale = np.dot(w, x) / unscaled.mean()
        return cls(
            mu + np.log(scale),
            h,
            w,
            lower=lower * scale,
            upper=upper * scale,
            seed=seed,
        )

    def _z(self, x):
        """Each component's standard-normal coordinate of x, in the last axis."""
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (np.log(np.maximum(x, 0.0))[..., None] - self.mu) / self.sigma

    def pdf(self, x):
        """The density at x."""
        x = np.asarray(x, dtype=float)
        z = self._z(x)
        inside = (x >= self.lower) & (x <= self.upper) & (x > 0.0)
        dens = (
            np.exp(-0.5 * z * z) * self._inv_mass / (np.sqrt(2.0 * np.pi) * self.sigma)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.dot(np.nan_to_num(dens), self.weights) / x
        return np.where(inside, out, 0.0)

    def cdf(self, x):
        """The cumulative distribution function at x."""
        z = np.clip(self._z(x), self._bot, self._top)
        return np.dot(_normal_mass(self._bot, z) * self._inv_mass, self.weights)

    def ppf(self, q):
        """
        The quantile function at q, by bisection in the log between the
        components' own quantiles, which bracket the mixture's.
        """
        q = np.asarray(q, dtype=float)
        qc = np.clip(q, 0.0, 1.0)[..., None]
        z = _truncated_normal_ppf(qc, self._bot, self._top)
        with np.errstate(invalid="ignore"):
            logs = np.where(self.weights > 0.0, self.mu + self.sigma * z, np.nan)
        inner = (q > 0.0) & (q < 1.0)
        lo = np.where(inner, np.nanmin(logs, axis=-1), 0.0)
        hi = np.where(inner, np.nanmax(logs, axis=-1), 0.0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = self.cdf(np.exp(mid)) < q
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
            if np.all(
                hi - lo <= 4.0 * np.finfo(float).eps * np.maximum(np.abs(hi), 1.0)
            ):
                break
        out = np.exp(0.5 * (lo + hi))
        out = np.where(q <= 0.0, self.lower, np.where(q >= 1.0, self.upper, out))
        return out[()] if out.ndim == 0 else out

    def partial_moment(self, order, lo, hi):
        """E[X**order 1{lo < X < hi}], in closed form."""
        n = float(order)
        zl = np.clip(self._z(lo), self._bot, self._top) - n * self.sigma
        zh = np.clip(self._z(hi), self._bot, self._top) - n * self.sigma
        scale = (self.weights * self._inv_mass) * np.exp(
            n * self.mu + 0.5 * (n * self.sigma) ** 2
        )
        return np.dot(_normal_mass(zl, zh), scale)

    def moment(self, order):
        """E[X**order], in closed form."""
        return float(self.partial_moment(order, self.lower, self.upper))

    def mean(self):
        """E[X]."""
        return self.moment(1)

    def draw(self, N):
        """
        N draws: a component for each by the weights, then that truncated
        lognormal by its quantile function.

        Parameters
        ----------
        N : int
            Number of draws.

        Returns
        -------
        np.array
        """
        k = self._rng.choice(self.mu.size, size=N, p=self.weights)
        z = _truncated_normal_ppf(self._rng.uniform(size=N), self._bot[k], self._top[k])
        return np.exp(self.mu[k] + self.sigma[k] * z)

    def _approx_equiprobable(self, N, endpoints=False):
        """
        A discrete approximation with N atoms of probability 1/N, each the
        conditional mean of its bin, so the mean is exact.

        Parameters
        ----------
        N : int
            Number of atoms.
        endpoints : bool, optional
            Whether to add zero-probability atoms at the infimum and supremum,
            by default False.

        Returns
        -------
        DiscreteDistribution
        """
        cuts = self.ppf(np.arange(N + 1) / N)
        pmv = np.ones(N) / N
        atoms = N * self.partial_moment(1, cuts[:-1], cuts[1:])
        if endpoints:
            atoms = np.r_[self.infimum, atoms, self.supremum]
            pmv = np.r_[0.0, pmv, 0.0]
        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}
        return DiscreteDistribution(pmv, atoms, seed=self.random_seed(), limit=limit)

    def _approx_components(self, N, endpoints=False):
        """
        A discrete approximation component by component: each component with
        weight discretized into N atoms of equal probability, the conditional
        means of its bins, each carrying its component's weight over N, so the
        mean is exact. The atoms are sorted.

        Parameters
        ----------
        N : int
            Number of atoms per component.
        endpoints : bool, optional
            Whether to add zero-probability atoms at the infimum and supremum,
            by default False.

        Returns
        -------
        DiscreteDistribution
        """
        parts = [
            (w, part.discretize(N, method="equiprobable"))
            for w, part in zip(self.weights, self.components)
            if w > 0.0
        ]
        atoms = np.concatenate([approx.atoms[0] for _, approx in parts])
        pmv = np.concatenate([w * approx.pmv for w, approx in parts])
        order = np.argsort(atoms, kind="stable")
        atoms, pmv = atoms[order], pmv[order]
        if endpoints:
            atoms = np.r_[self.infimum, atoms, self.supremum]
            pmv = np.r_[0.0, pmv, 0.0]
        limit = {"dist": self, "method": "components", "N": N, "endpoints": endpoints}
        return DiscreteDistribution(pmv, atoms, seed=self.random_seed(), limit=limit)


# The names these classes had before: they still work, with a DeprecationWarning
Lognormal = _renamed_class(LogNormal, "Lognormal")
MeanOneLogNormal = _renamed_class(LogNormalMeanOne, "MeanOneLogNormal")
TruncatedLognormal = _renamed_class(LogNormalTruncated, "TruncatedLognormal")
TruncatedLogNormal = _renamed_class(LogNormalTruncated, "TruncatedLogNormal")
MeanOneTruncatedLogNormal = _renamed_class(
    LogNormalTruncatedMeanOne, "MeanOneTruncatedLogNormal"
)


class Uniform(ContinuousFrozenDistribution):
    """
    A Uniform distribution.

    Parameters
    ----------
    bot : float or [float]
        One or more bottom values.
        Number of elements T in mu determines number
        of rows of output.
    top : float or [float]
        One or more top values.
        Number of elements T in top determines number of
        rows of output.
    seed : int
        Seed for random number generator.
    """

    def __init__(self, bot=0.0, top=1.0, seed=None):
        self.bot = np.asarray(bot)
        self.top = np.asarray(top)

        super().__init__(uniform, loc=self.bot, scale=self.top - self.bot, seed=seed)
        self.infimum = np.array(self.bot, dtype=float, ndmin=1)
        self.supremum = np.array(self.top, dtype=float, ndmin=1)

    def _approx_equiprobable(self, N, endpoints=False):
        """
        Makes a discrete approximation to this uniform distribution.

        Parameters
        ----------
        N : int
            The number of points in the discrete approximation.
        endpoints : bool
            Whether to include the endpoints in the approximation.

        Returns
        -------
        d : DiscreteDistribution
            Probability associated with each point in array of discrete
            points for discrete probability mass function.
        """
        pmv = np.ones(N) / float(N)

        center = (self.top + self.bot) / 2.0
        width = (self.top - self.bot) / 2.0
        atoms = center + width * np.linspace(-(N - 1.0) / 2.0, (N - 1.0) / 2.0, N) / (
            N / 2.0
        )

        if endpoints:  # insert endpoints with infinitesimally small mass
            atoms = np.concatenate(([self.bot], atoms, [self.top]))
            pmv = np.concatenate(([0.0], pmv, [0.0]))

        limit = {
            "dist": self,
            "method": "equiprobable",
            "N": N,
            "endpoints": endpoints,
        }

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self.random_seed(),
            limit=limit,
        )


class Weibull(ContinuousFrozenDistribution):
    """
    A Weibull distribution.

    Parameters
    ----------
    scale : float or [float]
        One or more scales.  Number of elements T in scale
        determines number of
        rows of output.
    shape : float or [float]
        One or more shape parameters. Number of elements T in scale
        determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    def __init__(self, scale=1.0, shape=1.0, seed=None):
        self.scale = np.asarray(scale)
        self.shape = np.asarray(shape)

        # Set up the RNG
        super().__init__(weibull_min, c=shape, scale=scale, seed=seed)
        self.infimum = np.array([0.0])
        self.supremum = np.array([np.inf])
