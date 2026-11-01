"""Simulations apply each period's time-varying parameters on entering the
period the solver assumes.

A cycle of four periods carries one change at index 2. In a life cycle
(``cycles == 1``) the solver pairs index t with the move from period t to
period t + 1, so it builds the change into ``solution[2]``, and the simulation
must apply it on entering period 3. Otherwise (``cycles == 0`` or
``cycles > 1``) the solver pairs index t with the move into period t (#1039),
so it builds the change into ``solution[1]``, and the simulation must apply it
on entering period 2.
"""

from copy import deepcopy

import numpy as np
import pytest

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsMarkovModel import (
    MarkovConsumerType,
    init_indshk_markov,
)
from HARK.distributions import DiscreteDistributionLabeled
from HARK.dual_measure import DualMeasureMixin
from HARK.utilities import get_arg_names

T = 4
CYCLES = [1, 0, 2]


class DualIndShock(DualMeasureMixin, IndShockConsumerType):
    pass


def entering(cycles):
    """The period on entering which the change at index 2 takes effect."""
    return 3 if cycles == 1 else 2


def looks_ahead(cycles):
    """The solution that the solver builds with the change at index 2."""
    return 2 if cycles == 1 else 1


def make_indshock(cycles, cls=IndShockConsumerType, **changes):
    params = dict(
        cycles=cycles,
        T_cycle=T,
        PermShkStd=[0.1] * T,
        TranShkStd=[0.1] * T,
        LivPrb=[1.0] * T,
        PermGroFac=[1.0] * T,
        Rfree=[1.03] * T,
        AgentCount=4000,
        T_sim=8,
        T_age=None,
        seed=0,
    )
    params.update(changes)
    return cls(**params)


def make_markov(cycles, MrkvArray, Rfree):
    """Two states with income fixed at one, as in test_ConsMarkovModel."""
    atom = DiscreteDistributionLabeled(
        pmv=np.array([1.0]),
        atoms=np.array([[1.0], [1.0]]),
        var_names=["PermShk", "TranShk"],
    )
    params = deepcopy(init_indshk_markov)
    params["constructors"] = dict(params["constructors"])
    params["constructors"]["IncShkDstn"] = None
    params["constructors"]["MrkvArray"] = None
    params.update(
        cycles=cycles,
        T_cycle=T,
        MrkvArray=MrkvArray,
        IncShkDstn=[[atom, atom] for _ in range(T)],
        Rfree=[np.array([R, R]) for R in Rfree],
        LivPrb=[np.ones(2) for _ in range(T)],
        PermGroFac=[np.ones(2) for _ in range(T)],
        MrkvPrbsInit=np.array([1.0, 0.0]),
        kLogInitMean=np.log(5.0),
        AgentCount=3,
        T_sim=T,
        T_age=None,
    )
    return MarkovConsumerType(**params)


def solve_recording(agent, name):
    """Solve, and return the value of ``name`` each solution of the first cycle
    was built with."""
    built_with = {}
    solver = agent.solve_one_period

    def recorder(**kwargs):
        solution = solver(**kwargs)
        built_with[id(solution)] = kwargs[name]
        return solution

    recorder.solver_args = get_arg_names(solver)
    agent.solve_one_period = recorder
    agent.solve()
    agent.solve_one_period = solver
    return [built_with[id(agent.solution[t])] for t in range(T)]


def applied_on_entering(agent, measure, steps=9):
    """Simulate agents who share t_cycle, and return measure(agent, before) for
    the first time each period is entered, before being the states the step
    started from."""
    agent.initialize_sim()
    seen = {}
    for step in range(steps):
        entered = np.unique(agent.t_cycle)
        before = {
            k: v.copy() for k, v in agent.state_now.items() if isinstance(v, np.ndarray)
        }
        agent.sim_one_period()
        # The first step draws everyone as a newborn, from index 0
        if step > 0 and entered.size == 1:
            seen.setdefault(int(entered[0]), measure(agent, before))
    return seen


@pytest.mark.parametrize("cycles", CYCLES)
@pytest.mark.parametrize(
    "name, values, changed",
    [
        ("PermGroFac", [1.0, 1.0, 2.0, 1.0], 2.0),
        ("Rfree", [1.03, 1.03, 1.5, 1.03], 1.5),
        ("LivPrb", [1.0, 1.0, 0.5, 1.0], 0.5),
    ],
)
def test_solver_builds_the_change_into_the_solution_that_looks_ahead(
    cycles, name, values, changed
):
    agent = make_indshock(cycles, **{name: values})
    built_with = solve_recording(agent, name)
    assert [t for t in range(T) if built_with[t] == changed] == [looks_ahead(cycles)]


@pytest.mark.parametrize("cycles", CYCLES)
def test_markov_solver_builds_the_switch_into_the_solution_that_looks_ahead(cycles):
    MrkvArray = [np.eye(2) for _ in range(T)]
    MrkvArray[2] = np.array([[0.0, 1.0], [0.0, 1.0]])
    agent = make_markov(cycles, MrkvArray, [1.02] * T)
    built_with = solve_recording(agent, "MrkvArray")
    switching = [t for t in range(T) if built_with[t][0, 1] == 1.0]
    assert switching == [looks_ahead(cycles)]


@pytest.mark.parametrize("cycles", CYCLES)
def test_growth_return_and_shocks_arrive_when_the_solver_assumes(cycles):
    agent = make_indshock(
        cycles,
        PermGroFac=[1.0, 1.0, 2.0, 1.0],
        Rfree=[1.03, 1.03, 1.5, 1.03],
        TranShkStd=[0.1, 0.1, 0.4, 0.1],
    )
    agent.solve()

    def measure(agent, before):
        saved = before["aNrm"] > 0.01
        employed = agent.shocks["TranShk"] > 0.31  # leaves out IncUnemp = 0.3
        growth = np.mean(agent.state_now["pLvl"] / before["pLvl"])
        R = np.median(
            agent.state_now["bNrm"][saved]
            * agent.shocks["PermShk"][saved]
            / before["aNrm"][saved]
        )
        sd = np.std(np.log(agent.shocks["TranShk"][employed]))
        return growth, R, sd

    seen = applied_on_entering(agent, measure)
    assert sorted(seen) == list(range(T))
    for k, (growth, R, sd) in seen.items():
        changed = k == entering(cycles)
        assert growth == pytest.approx(2.0 if changed else 1.0, rel=0.02), k
        assert R == pytest.approx(1.5 if changed else 1.03, rel=1e-9), k
        assert (sd > 0.3) if changed else (sd < 0.15), k


@pytest.mark.parametrize("cycles", CYCLES)
def test_deaths_are_drawn_with_the_survival_the_solver_assumes(cycles):
    agent = make_indshock(cycles, LivPrb=[1.0, 1.0, 0.5, 1.0], death_shuffle=True)
    agent.solve()
    agent.initialize_sim()
    agent.t_age = np.full(agent.AgentCount, 5)
    for k in range(T):
        agent.t_cycle = np.full(agent.AgentCount, k)  # the period being entered
        rate = np.mean(agent.sim_death())
        assert rate == (0.5 if k == entering(cycles) else 0.0), k


@pytest.mark.parametrize("cycles", CYCLES)
def test_markov_switch_and_return_arrive_when_the_solver_assumes(cycles):
    MrkvArray = [np.eye(2) for _ in range(T)]
    MrkvArray[2] = np.array([[0.0, 1.0], [0.0, 1.0]])
    Rfree = [1.02, 1.02, 1.5, 1.02]
    agent = make_markov(cycles, MrkvArray, Rfree)
    agent.solve()
    agent.track_vars = ["Mrkv", "aNrm", "mNrm", "TranShk", "PermShk"]
    agent.initialize_sim()
    agent.simulate()
    h = agent.history
    k = entering(cycles)
    np.testing.assert_array_equal(h["Mrkv"][:, 0], [0] * k + [1] * (T - k))
    for s in range(1, T):
        implied = (h["mNrm"][s] - h["TranShk"][s]) * h["PermShk"][s]
        implied /= h["aNrm"][s - 1]
        np.testing.assert_allclose(implied, Rfree[2] if s == k else 1.02, rtol=1e-12)


@pytest.mark.parametrize("cycles", CYCLES)
def test_dual_measure_draws_growth_with_p_when_the_solver_assumes(cycles):
    agent = make_indshock(cycles, cls=DualIndShock, PermGroFac=[1.0, 1.0, 2.0, 1.0])
    agent.solve()
    agent.setup_Q_measure()

    def measure(agent, before):
        return np.mean(agent.shocks["PermShk"]), np.mean(agent.shocks_Q["PermShk"])

    seen = applied_on_entering(agent, measure)
    assert sorted(seen) == list(range(T))
    for k, (p, q) in seen.items():
        growth = 2.0 if k == entering(cycles) else 1.0
        assert p == pytest.approx(growth, rel=0.02), k
        # Q reweights the permanent shock by E[psi^2]/E[psi], about 1.01 here
        assert q == pytest.approx(growth, rel=0.05), k


@pytest.mark.parametrize("cycles", CYCLES)
def test_model_file_simulator_applies_growth_when_the_solver_assumes(cycles):
    agent = make_indshock(cycles, PermGroFac=[1.0, 1.0, 2.0, 1.0])
    agent.solve()
    agent.track_vars = ["pLvl", "t_seq"]
    agent.initialize_sym()
    agent.symulate(T)
    h = agent.hystory
    pLvl = h["pLvl"].mean(axis=1)
    for s in range(1, T):
        period = int(np.unique(h["t_seq"][s])[0]) % T
        growth = pLvl[s] / pLvl[s - 1]
        expected = 2.0 if period == entering(cycles) else 1.0
        assert growth == pytest.approx(expected, rel=0.02), period
