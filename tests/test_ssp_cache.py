# -*- coding: utf-8 -*-
"""Tests that a new StellarPopulation keeps the cached SSPs only when it may.

All StellarPopulation objects in a process share one Fortran state. Each test
runs its scenario in a fresh Python process, so the state it starts from is
known, and compares the spectra with ones from other fresh processes. Each
scenario counts the calls to driver.set_ssp_params, which clears the cache.
Run this file as a script to run one scenario by hand.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

skip_slow_tests = pytest.mark.skipif(
    (sys.platform.startswith("win") or sys.platform.startswith("darwin"))
    and "CI" in os.environ,
    reason="Slow tests only run on Linux CI",
)

TAGE = 1.0

# A narrow LSF keeps the SSP smoothing cheap on high resolution libraries.
LSF_WAVE = np.linspace(5000.0, 5100.0, 11)
LSF_SIGMA = np.full(11, 100.0)


class _CountingDriver(object):
    """Wrap the Fortran driver and count the calls to set_ssp_params."""

    def __init__(self, driver):
        self._driver = driver
        self.n_set_ssp_params = 0

    def set_ssp_params(self, *args):
        self.n_set_ssp_params += 1
        return self._driver.set_ssp_params(*args)

    def __getattr__(self, name):
        return getattr(self._driver, name)


def _counting_driver():
    import fsps.fsps

    if not isinstance(fsps.fsps.driver, _CountingDriver):
        fsps.fsps.driver = _CountingDriver(fsps.fsps.driver)
    return fsps.fsps.driver


def _spectrum(sp, **kwargs):
    """Return the spectrum at TAGE, the seconds it took and the set_ssp_params calls."""
    drv = _counting_driver()
    n = drv.n_set_ssp_params
    t = time.time()
    spec = sp.get_spectrum(tage=TAGE, **kwargs)[1]
    return spec, time.time() - t, drv.n_set_ssp_params - n


def _new(**kwargs):
    from fsps import StellarPopulation

    _counting_driver()
    return StellarPopulation(zcontinuous=0, **kwargs)


# Scenarios. Each runs in a fresh process and returns a dict of arrays.


def scenario_reference(kind):
    """Return the spectrum of the first population in a fresh process."""
    if kind == "default":
        sp = _new()
    elif kind == "imf1":
        sp = _new(imf_type=1)
    elif kind == "nosv":
        sp = _new(smooth_velocity=False)
    elif kind == "lsf1":
        sp = _new(smooth_lsf=True)
        sp.set_lsf(LSF_WAVE, LSF_SIGMA)
    elif kind == "lsf2":
        sp = _new(smooth_lsf=True)
        sp.set_lsf(LSF_WAVE, 3 * LSF_SIGMA)
    elif kind == "dust":
        sp = _new(dust2=0.5)
    else:
        raise ValueError(kind)
    spec, dt, n = _spectrum(sp)
    return dict(spec=spec, dt=dt, n=n)


def scenario_same_params():
    """A second population with the same parameters keeps the cache."""
    s1, dt1, n1 = _spectrum(_new())
    s2, dt2, n2 = _spectrum(_new())
    # a different CSP parameter only
    s3, dt3, n3 = _spectrum(_new(dust2=0.5))
    return dict(s1=s1, s2=s2, s3=s3, dt=[dt1, dt2, dt3], n=[n1, n2, n3])


def scenario_imf_switch():
    """A population with another imf_type clears the cache, and so does going back."""
    s1, dt1, n1 = _spectrum(_new())
    s2, dt2, n2 = _spectrum(_new(imf_type=1))
    s3, dt3, n3 = _spectrum(_new())
    # a new population that matches the cache, then changes its own SSP parameter
    sp = _new()
    s4, dt4, n4 = _spectrum(sp)
    sp.params["imf_type"] = 1
    s5, dt5, n5 = _spectrum(sp)
    return dict(
        s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
        dt=[dt1, dt2, dt3, dt4, dt5], n=[n1, n2, n3, n4, n5],
    )  # fmt: skip


def scenario_lsf():
    """set_lsf clears the cache key, also when called on another population."""
    a = _new(smooth_lsf=True)
    a.set_lsf(LSF_WAVE, LSF_SIGMA)
    s1, dt1, n1 = _spectrum(a)
    # same inputs, LSF unchanged
    s2, dt2, n2 = _spectrum(_new(smooth_lsf=True))
    # another population sets a new LSF but does not use it itself
    c = _new()
    c.set_lsf(LSF_WAVE, 3 * LSF_SIGMA)
    s3, dt3, n3 = _spectrum(_new(smooth_lsf=True))
    return dict(s1=s1, s2=s2, s3=s3, dt=[dt1, dt2, dt3], n=[n1, n2, n3])


def scenario_smooth_velocity():
    """smooth_velocity changes the SSPs (AGB dust shells) and is part of the key."""
    a = _new()
    s1, dt1, n1 = _spectrum(a)
    s2, dt2, n2 = _spectrum(_new(smooth_velocity=False))
    s3, dt3, n3 = _spectrum(_new())
    # a population changes smooth_velocity, a CSP parameter, without a reset
    a.params["smooth_velocity"] = False
    s4, dt4, n4 = _spectrum(a)
    a.params["smooth_velocity"] = True
    s5, dt5, n5 = _spectrum(a)
    # the cache now holds SSPs built with smooth_velocity False, so a new one resets
    s6, dt6, n6 = _spectrum(_new())
    return dict(
        s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6,
        dt=[dt1, dt2, dt3, dt4, dt5, dt6], n=[n1, n2, n3, n4, n5, n6],
    )  # fmt: skip


def _child_spectrum(queue):
    s, dt, n = _spectrum(_new())
    queue.put((s, dt, n))


def scenario_fork():
    """A forked child keeps the SSPs it inherits from its parent."""
    import multiprocessing as mp

    s1, dt1, n1 = _spectrum(_new())
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child_spectrum, args=(queue,))
    proc.start()
    s2, dt2, n2 = queue.get()
    proc.join()
    assert proc.exitcode == 0
    return dict(s1=s1, s2=s2, dt=[dt1, dt2], n=[n1, n2])


def scenario_interleave():
    """Two populations with different imf_type, used in turn.

    Each turn marks the population fully dirty, since the other one changed the
    shared Fortran state. Two more populations differ in a CSP parameter only.
    """
    a, b = _new(), _new(imf_type=1)
    c, d = _new(), _new(dust2=0.5)
    out = dict(a=[], b=[], c=[], d=[], n=[], dt=[])
    for _ in range(2):
        for name, sp in (("a", a), ("b", b)):
            sp.params.dirtiness = 2
            s, dt, n = _spectrum(sp)
            out[name].append(s)
            out["n"].append(n)
            out["dt"].append(dt)
    for _ in range(2):
        for name, sp in (("c", c), ("d", d)):
            sp.params.dirtiness = 2
            s, dt, n = _spectrum(sp)
            out[name].append(s)
            out["n"].append(n)
            out["dt"].append(dt)
    return out


def _run(tmp_path, name, *args):
    """Run one scenario in a fresh process and return its arrays."""
    out = str(tmp_path / ("_".join((name,) + args) + ".npz"))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    # macOS stops a forked child that uses Objective-C classes without this
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    cmd = [sys.executable, os.path.abspath(__file__), name, out] + list(args)
    subprocess.run(cmd, check=True, env=env)
    with np.load(out) as f:
        return {k: f[k] for k in f.files}


def _equal(x, y):
    return x.shape == y.shape and x.tobytes() == y.tobytes()


def test_cache_inputs():
    """The key compares exact types and bits, and is None when ssp_gen reads more."""
    from fsps.fsps import ParameterSet, _ssp_cache_inputs

    def key(**kwargs):
        params = dict(imf_type=2, smooth_lsf=False, smooth_velocity=True)
        params.update(add_neb_emission=False, dell=0.0)
        params.update(kwargs)
        ps = ParameterSet(**params)
        ps.ssp_params = ["imf_type", "smooth_lsf", "dell"]
        return _ssp_cache_inputs(ps)

    base = key()
    assert None not in base
    assert key() == base
    assert key(dell=np.float64(0.0)) != base
    assert key(dell=-0.0) != base
    assert key(dell=np.array([0.0])) != base
    assert key(smooth_lsf=0) != base
    assert key(smooth_velocity=1)[1] != base[1]
    assert key(add_neb_emission=True) == base
    assert key(add_neb_emission=np.int64(1)) == base
    assert key(add_neb_emission=2)[1] is None
    assert key(add_neb_emission=1.0)[1] is None
    assert key(imf_type=5)[0] is None
    assert key(imf_type=2.0)[0] is None
    assert key(dell=[0.0])[0] is None
    assert key(dell=float("nan"))[0] == key(dell=float("nan"))[0]


@skip_slow_tests
def test_same_params_keeps_cache(tmp_path):
    r = _run(tmp_path, "same_params")
    assert list(r["n"]) == [1, 0, 0]
    assert r["dt"][1] < 1.0 and r["dt"][2] < 1.0
    assert _equal(r["s1"], r["s2"])
    assert _equal(r["s3"], _run(tmp_path, "reference", "dust")["spec"])


@skip_slow_tests
def test_imf_switch_resets(tmp_path):
    r = _run(tmp_path, "imf_switch")
    assert list(r["n"]) == [1, 1, 1, 0, 1]
    ref1 = _run(tmp_path, "reference", "imf1")["spec"]
    assert not _equal(r["s1"], ref1)
    assert _equal(r["s2"], ref1)
    assert _equal(r["s3"], r["s1"])
    assert _equal(r["s4"], r["s1"])
    assert _equal(r["s5"], ref1)


@skip_slow_tests
def test_lsf_invalidates(tmp_path):
    r = _run(tmp_path, "lsf")
    assert list(r["n"]) == [1, 0, 1]
    ref1 = _run(tmp_path, "reference", "lsf1")["spec"]
    ref2 = _run(tmp_path, "reference", "lsf2")["spec"]
    assert not _equal(ref1, ref2)
    assert _equal(r["s1"], ref1)
    assert _equal(r["s2"], ref1)
    assert _equal(r["s3"], ref2)


@skip_slow_tests
def test_smooth_velocity_invalidates(tmp_path):
    r = _run(tmp_path, "smooth_velocity")
    assert list(r["n"]) == [1, 1, 1, 0, 0, 1]
    ref = _run(tmp_path, "reference", "nosv")["spec"]
    assert not _equal(r["s1"], ref)
    assert _equal(r["s2"], ref)
    assert _equal(r["s3"], r["s1"])
    assert _equal(r["s6"], r["s1"])


@skip_slow_tests
def test_fork_keeps_cache(tmp_path):
    if sys.platform.startswith("win"):
        pytest.skip("fork is not available")
    r = _run(tmp_path, "fork")
    assert list(r["n"]) == [1, 0]
    assert r["dt"][1] < 1.0
    assert _equal(r["s1"], r["s2"])


@skip_slow_tests
def test_interleave(tmp_path):
    r = _run(tmp_path, "interleave")
    # a and b reset on every turn. c resets once, since b built the cache, and
    # then c and d share it.
    assert list(r["n"]) == [1, 1, 1, 1, 1, 0, 0, 0]
    ref = _run(tmp_path, "reference", "default")["spec"]
    ref1 = _run(tmp_path, "reference", "imf1")["spec"]
    refd = _run(tmp_path, "reference", "dust")["spec"]
    for s in r["a"]:
        assert _equal(s, ref)
    for s in r["b"]:
        assert _equal(s, ref1)
    for s in r["c"]:
        assert _equal(s, ref)
    for s in r["d"]:
        assert _equal(s, refd)


if __name__ == "__main__":
    name, out, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    result = globals()["scenario_" + name](*args)
    np.savez(out, **{k: np.asarray(v) for k, v in result.items()})
