"""UpdateDeduplicator: bounded, LRU-fresh memory of processed update ids."""

import pytest

from src.telegram.dedup import UpdateDeduplicator


def test_first_sighting_registers_and_reports_false():
    dedup = UpdateDeduplicator(capacity=10)

    assert dedup.seen_before(1) is False
    assert dedup.seen_before(1) is True


def test_distinct_ids_are_tracked_independently():
    dedup = UpdateDeduplicator(capacity=10)

    assert dedup.seen_before(1) is False
    assert dedup.seen_before(2) is False
    assert dedup.seen_before(1) is True
    assert dedup.seen_before(2) is True


def test_overflow_evicts_the_oldest_id():
    dedup = UpdateDeduplicator(capacity=2)
    dedup.seen_before(1)
    dedup.seen_before(2)

    dedup.seen_before(3)  # evicts 1

    assert dedup.seen_before(1) is False  # forgotten, registered anew
    assert dedup.seen_before(3) is True


def test_re_seeing_an_id_keeps_it_hot_while_others_evict():
    dedup = UpdateDeduplicator(capacity=2)
    dedup.seen_before(1)
    dedup.seen_before(2)

    assert dedup.seen_before(1) is True  # moved to the fresh end
    dedup.seen_before(3)  # now evicts 2, not 1

    assert dedup.seen_before(1) is True
    assert dedup.seen_before(2) is False


@pytest.mark.parametrize("capacity", [0, -1])
def test_capacity_below_one_is_rejected(capacity):
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        UpdateDeduplicator(capacity=capacity)
