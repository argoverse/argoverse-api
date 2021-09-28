import numpy as np

from argoverse.utils.metric_time import TimeUnit, to_metric_time


def test_nanoseconds_to_seconds():
    """Test conversion from nanoseconds to seconds."""
    ts_ns = 950000000
    ts_s = to_metric_time(ts=ts_ns, src=TimeUnit.Nanosecond, dst=TimeUnit.Second)
    assert np.isclose(ts_s, 0.95)
    assert np.isclose(ts_s, ts_ns / 1e9)


def test_seconds_to_nanoseconds():
    """Test conversion from seconds to nanoseconds."""
    ts_s = 0.95
    ts_ns = to_metric_time(ts=ts_s, src=TimeUnit.Second, dst=TimeUnit.Nanosecond)
    assert np.isclose(ts_ns, 950000000)
    assert np.isclose(ts_ns, ts_s * 1e9)


def test_milliseconds_to_seconds():
    """Test conversion from milliseconds to seconds."""
    ts_ms = 300
    ts_s = to_metric_time(ts=ts_ms, src=TimeUnit.Millisecond, dst=TimeUnit.Second)
    assert np.isclose(ts_s, 0.3)


def test_seconds_to_milliseconds():
    """Test conversion from seconds to milliseconds."""
    ts_s = 0.3
    ts_ms = to_metric_time(ts=ts_s, src=TimeUnit.Second, dst=TimeUnit.Millisecond)
    assert np.isclose(ts_ms, 300)
