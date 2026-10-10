# -*- coding: utf-8 -*-
"""Tests for ``StellarPopulation.build_ssps``.

Each SSP slot takes tens of seconds to build, so these tests build only a few
slots near solar metallicity. The reference SSPs and spectra are built the
normal way in processes forked before any build in this module.
"""

import multiprocessing
import threading
import time

import numpy as np
import pytest

from fsps import StellarPopulation
from fsps.fsps import driver

pytestmark = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="build_ssps forks helper processes",
)


def _grid():
    """Return three metallicity indices around solar and log Z between them."""
    sp = StellarPopulation(zcontinuous=1)
    logz = np.log10(sp.zlegend / sp.solar_metallicity)
    k = int(np.clip(np.argmin(np.abs(logz)), 1, len(logz) - 2))
    zmets = [k, k + 1, k + 2]  # 1-based indices around the solar index k + 1
    lz = logz[k - 1 : k + 2]
    # points strictly between grid values, so FSPS uses only these slots
    logzsols = [
        0.5 * (lz[0] + lz[1]),
        lz[1] + 0.3 * (lz[2] - lz[1]),
        0.5 * (lz[1] + lz[2]),
    ]
    return zmets, logzsols


def _reference(conn, overrides, zmets, logzsols):
    """Compute spectra and SSPs the normal way and send them through ``conn``."""
    sp = StellarPopulation(zcontinuous=1, **overrides)
    # send the SSP parameters, which get_spectrum would do on its first call
    sp._update_params()
    spectra = []
    for logzsol in logzsols:
        sp.params["logzsol"] = logzsol
        spectra.append(sp.get_spectrum(tage=0)[1])
    ns, nt = driver.get_nspec(), driver.get_ntfull()
    slots = {}
    for zi in zmets:
        # FSPS builds the slot here if no spectrum above needed it
        slots[zi] = driver.get_ssp_slot(ns, nt, zi, 1)
    conn.send((slots, spectra))
    conn.close()


class _References:
    """Reference SSPs, computed in forked processes started on creation."""

    def __init__(self, cases):
        ctx = multiprocessing.get_context("fork")
        self._procs, self._readers, self._results = {}, {}, {}
        for name, (overrides, zmets, logzsols) in cases.items():
            reader, writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_reference, args=(writer, overrides, zmets, logzsols)
            )
            proc.start()
            writer.close()
            self._procs[name], self._readers[name] = proc, reader

    def __getitem__(self, name):
        if name not in self._results:
            self._results[name] = self._readers[name].recv()
            self._procs[name].join()
            assert self._procs[name].exitcode == 0
        return self._results[name]

    def close(self):
        for name, proc in self._procs.items():
            if name not in self._results:
                proc.terminate()
            proc.join()
            self._readers[name].close()


@pytest.fixture(scope="module")
def grid():
    return _grid()


@pytest.fixture(scope="module")
def references(grid):
    zmets, logzsols = grid
    refs = _References(
        {
            "kroupa": ({}, zmets, logzsols),
            "chabrier": ({"imf_type": 1}, zmets[:2], []),
            "salpeter": ({"imf_type": 0}, zmets[:2], []),
        }
    )
    yield refs
    refs.close()


def _stored_slots(zmets):
    """Return the stored slots, checking that none is built on reading."""
    ns, nt = driver.get_nspec(), driver.get_ntfull()
    t0 = time.time()
    slots = {zi: driver.get_ssp_slot(ns, nt, zi, 1) for zi in zmets}
    # reading a stored slot is a copy, building one takes many seconds
    assert time.time() - t0 < 2.0
    return slots


def _assert_slots_equal(got, expected):
    for zi in expected:
        for a, b in zip(got[zi], expected[zi]):
            assert np.array_equal(a, b)


def _spectrum_in_child(conn, sp, logzsol):
    sp.params["logzsol"] = logzsol
    conn.send(sp.get_spectrum(tage=1.0)[1])
    conn.close()


def test_build_ssps_parallel(references, grid):
    """Slots built by three processes, and spectra made from them, equal those
    built normally. Afterwards no threads or helpers are left, and this process
    can fork."""
    zmets, logzsols = grid
    sp = StellarPopulation(zcontinuous=1)
    threads = threading.active_count()
    t0 = time.time()
    sp.build_ssps(zmet=zmets, afeindx=1, processes=3)
    print(f"build_ssps of 3 slots with 3 processes: {time.time() - t0:.1f} s")
    assert threading.active_count() == threads

    slots, spectra = references["kroupa"]
    _assert_slots_equal(_stored_slots(zmets), slots)
    for logzsol, expected in zip(logzsols, spectra):
        sp.params["logzsol"] = logzsol
        assert np.array_equal(sp.get_spectrum(tage=0)[1], expected)

    ctx = multiprocessing.get_context("fork")
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_spectrum_in_child, args=(writer, sp, logzsols[1]))
    proc.start()
    writer.close()
    child = reader.recv()
    proc.join(timeout=300)
    reader.close()
    assert proc.exitcode == 0
    sp.params["logzsol"] = logzsols[1]
    assert np.array_equal(child, sp.get_spectrum(tage=1.0)[1])


def test_build_ssps_serial(references, grid, monkeypatch):
    """With processes=1 every slot is built in this process without forking."""
    zmets, _ = grid

    def no_fork(*args, **kwargs):
        raise AssertionError("build_ssps forked with processes=1")

    monkeypatch.setattr(multiprocessing, "get_context", no_fork)
    sp = StellarPopulation(zcontinuous=1, imf_type=0)
    t0 = time.time()
    sp.build_ssps(zmet=zmets[:2], afeindx=1, processes=1)
    print(f"build_ssps of 2 slots with 1 process: {time.time() - t0:.1f} s")
    _assert_slots_equal(_stored_slots(zmets[:2]), references["salpeter"][0])


def test_build_ssps_ssp_params(references, grid):
    """Slots are built with the SSP parameters of the calling object."""
    zmets, _ = grid
    sp = StellarPopulation(zcontinuous=1, imf_type=1)
    t0 = time.time()
    sp.build_ssps(zmet=zmets[:2], afeindx=1, processes=2)
    print(f"build_ssps of 2 slots with 2 processes: {time.time() - t0:.1f} s")
    slots = _stored_slots(zmets[:2])
    _assert_slots_equal(slots, references["chabrier"][0])
    kroupa = references["kroupa"][0]
    assert not np.array_equal(slots[zmets[0]][0], kroupa[zmets[0]][0])


def test_build_ssps_other_object(references, grid):
    """build_ssps sends this object's SSP parameters even if they are unchanged
    since its last spectrum and another object changed the ones FSPS holds."""
    zmets, _ = grid
    sp = StellarPopulation(zcontinuous=1)
    sp._update_params()
    other = StellarPopulation(zcontinuous=1, imf_type=1)
    other._update_params()
    assert sp.params.dirtiness == 0
    sp.build_ssps(zmet=zmets[1], afeindx=1, processes=1)
    expected = references["kroupa"][0]
    _assert_slots_equal(_stored_slots([zmets[1]]), {zmets[1]: expected[zmets[1]]})


def test_build_ssps_arguments(monkeypatch):
    """Bad arguments raise before anything is built."""
    sp = StellarPopulation(zcontinuous=1)
    nz, nafe = len(sp.zlegend), sp.n_afe
    for kwargs in [
        {"zmet": 0},
        {"zmet": nz + 1},
        {"zmet": 1.5},
        {"zmet": []},
        {"afeindx": nafe + 1},
        {"processes": 0},
        {"processes": 1.5},
    ]:
        with pytest.raises(ValueError):
            sp.build_ssps(**kwargs)
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    with pytest.raises(RuntimeError, match="fork"):
        sp.build_ssps(zmet=[1, 2], afeindx=1, processes=2)
