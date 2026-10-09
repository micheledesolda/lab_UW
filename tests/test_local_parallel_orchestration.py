from types import SimpleNamespace

from local_optimization_FWI_velocity_1D import (
    _build_local_inversion_lanes,
    resolve_worker_count,
)


def test_interlaced_lanes_cover_each_record_once():
    rows = [SimpleNamespace(Index=float(index)) for index in range(31)]

    lanes = _build_local_inversion_lanes(
        paired_rows=rows,
        remaining_positions=list(range(1, 31)),
        anchor_stride=10,
    )

    assert len(lanes) == 10
    assert lanes[0] == [1, 11, 21]
    assert lanes[-1] == [10, 20, 30]
    assert sorted(position for lane in lanes for position in lane) == list(
        range(1, 31)
    )


def test_invalid_mechanical_matches_are_not_scheduled():
    rows = [
        SimpleNamespace(Index=0.0),
        SimpleNamespace(Index=float("nan")),
        SimpleNamespace(Index=2.0),
        SimpleNamespace(Index=3.0),
    ]

    lanes = _build_local_inversion_lanes(
        paired_rows=rows,
        remaining_positions=[1, 2, 3],
        anchor_stride=2,
    )

    assert sorted(position for lane in lanes for position in lane) == [2, 3]


def test_local_worker_limit_is_independent_from_global_limit():
    params = {
        "parallel_cpu_reserve": 1,
        "global_search_workers": 8,
        "local_inversion_workers": 2,
    }

    assert resolve_worker_count(
        params,
        "local_inversion_workers",
        number_of_tasks=10,
    ) <= 2

