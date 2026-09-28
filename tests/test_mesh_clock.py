"""shard/mesh/clock.py: virtual time must never move while work is pending (BUILD-PLAN F4).

The fault harness trusts VirtualClock to turn 2 s deadlines into nothing WITHOUT letting a
deadline beat a message that is still on a socket, in a queue or being computed. These tests pin
that rule and its escape hatch (Stuck instead of a silent hang).

Run: python3 -m pytest tests/test_mesh_clock.py -q
"""
import threading
import time

import pytest

from shard.mesh.clock import Closed, RealClock, Stuck, Timeout, VirtualClock


@pytest.fixture
def vc():
    c = VirtualClock(stuck_after=1.0)
    c.adopt("test")
    yield c
    c.disown()


def test_time_moves_only_at_quiescence(vc):
    mb = vc.mailbox("main")

    def sleeper():
        vc.sleep(5.0)
        mb.put(vc.now())

    vc.spawn("sleeper", sleeper)
    time.sleep(0.1)                               # main is busy (real work): time is frozen
    assert vc.now() == 0.0
    assert mb.get(timeout=10.0) == 5.0            # main idles -> time jumps to the sleeper's timer
    assert vc.now() == 5.0


def test_a_deadline_never_beats_an_in_flight_message(vc):
    """A message sent on a stream holds time still until the reader takes it, however long the
    real transit takes."""
    mb = vc.mailbox("main")
    acct = vc.conn("a->b")
    acct.sent()

    def reader():
        with vc.idle():
            time.sleep(0.3)                       # a slow, real recv
        acct.took()
        mb.put("reply")

    vc.spawn("reader", reader)
    assert mb.get(timeout=1.0) == "reply"         # would be Timeout if time advanced during the recv
    assert vc.now() == 0.0


def test_a_dead_stream_releases_its_messages(vc):
    mb = vc.mailbox("main")
    acct = vc.conn("a->b")
    acct.sent()
    acct.sent()
    vc.spawn("killer", lambda: (time.sleep(0.1), acct.dead()))
    with pytest.raises(Timeout):
        mb.get(timeout=2.0)
    assert vc.now() == 2.0                        # the deadline fired exactly, after the release


def test_took_after_dead_counts_nothing(vc):
    acct = vc.conn("x")
    acct.sent()
    acct.dead()
    acct.took()                                   # the reader got the bytes after the stream died
    vc.sleep(1.0)                                 # still advances: load is only the main actor
    assert vc.now() == 1.0


def test_put_wakes_the_consumer_before_time_can_move(vc):
    """The consumer is marked busy by put(), so a timer due later cannot fire in between."""
    mb, out = vc.mailbox("c"), vc.mailbox("main")

    def consumer():
        x = mb.get()
        time.sleep(0.2)                           # real work, still at the put's instant
        out.put((x, vc.now()))

    vc.spawn("consumer", consumer)
    vc.spawn("late", lambda: (vc.sleep(0.5), out.put(("late", vc.now()))))
    vc.settle()
    mb.put("job")
    assert out.get(timeout=10) == ("job", 0.0)
    assert out.get(timeout=10) == ("late", 0.5)


def test_equal_deadlines_fire_together_and_in_order(vc):
    out = vc.mailbox("main")
    for i, dt in enumerate((0.3, 0.1, 0.2, 0.1)):
        vc.spawn(f"s{i}", lambda i=i, dt=dt: (vc.sleep(dt), out.put((round(vc.now(), 9), i))))
    got = sorted(out.get(timeout=5) for _ in range(4))
    assert got == [(0.1, 1), (0.1, 3), (0.2, 2), (0.3, 0)]


def test_settle_does_not_move_time(vc):
    done = []
    vc.spawn("worker", lambda: (time.sleep(0.1), done.append(1)))
    vc.spawn("timer", lambda: vc.sleep(3.0))
    vc.settle()
    assert done == [1] and vc.now() == 0.0


def test_close_wakes_the_consumer(vc):
    mb = vc.mailbox("m")
    res = vc.mailbox("main")

    def consumer():
        try:
            mb.get()
        except Closed:
            res.put("closed")

    vc.spawn("consumer", consumer)
    vc.settle()
    mb.close()
    assert res.get(timeout=1) == "closed"
    assert not mb.put(1)


def test_stuck_is_loud_not_a_hang(vc):
    """Nothing can ever wake the waiter: a Stuck error with a dump, not an endless wait."""
    t0 = time.monotonic()
    with pytest.raises(Stuck) as ei:
        vc.mailbox("never").get()
    assert time.monotonic() - t0 < 10
    assert "waiting: test" in str(ei.value)


def test_a_busy_actor_blocked_outside_the_clock_is_reported(vc):
    gate = threading.Event()
    vc.spawn("wedged", gate.wait)                 # busy forever: time can never move
    with pytest.raises(Stuck) as ei:
        vc.mailbox("m").get(timeout=5.0)
    assert "wedged" in str(ei.value)
    gate.set()


def test_unregistered_threads_are_refused(vc):
    mb = vc.mailbox("m")
    err = []

    def rogue():
        try:
            mb.get(timeout=1)
        except RuntimeError as e:
            err.append(str(e))

    t = threading.Thread(target=rogue)
    t.start()
    t.join(5)
    assert err and "not a VirtualClock actor" in err[0]


def test_real_clock_mailbox():
    rc = RealClock()
    mb = rc.mailbox()
    t0 = time.monotonic()
    with pytest.raises(Timeout):
        mb.get(timeout=0.05)
    assert time.monotonic() - t0 >= 0.045
    rc.spawn("p", lambda: (time.sleep(0.02), mb.put(7)))
    assert mb.get(timeout=2) == 7
    with pytest.raises(Timeout):
        mb.get(timeout=0)
    mb.close()
    with pytest.raises(Closed):
        mb.get()


def test_a_livelock_is_stuck_too(vc):
    """Two actors answering each other forever at one instant keep load above 0: time can
    never move, and the waiter must hear about it (the node ack ping-pong the review found)."""
    a, b = vc.mailbox("a"), vc.mailbox("b")
    stop = threading.Event()

    def pong(inbox, other):
        while not stop.is_set():
            try:
                x = inbox.get(timeout=0.5)
            except (Timeout, Closed):
                return
            other.put(x)

    vc.spawn("ping", pong, a, b)
    vc.spawn("pong", pong, b, a)
    a.put(1)
    with pytest.raises(Stuck):
        vc.mailbox("main").get(timeout=1.0)
    stop.set()
    a.close()
    b.close()
