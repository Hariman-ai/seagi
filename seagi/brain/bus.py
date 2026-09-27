"""Event bus — dispatches events to subscribed capabilities.

In v2 capabilities communicate ONLY through this bus.  No
direct calls between capabilities.  This decouples them and
makes parallelism implementable later without refactoring
every call site.

Phase 1: synchronous in-process dispatch.  Simple.  Sufficient
for skeleton + Thalamic Gate.

Phase 3+: same interface, but implementation routes through a
real concurrent runtime (process queues / thread-per-capability
/ actors).  Subscribers don't change — their `handle(event)`
method just runs in a different execution context.

Subscriber contract
-------------------
Any capability that wants to receive events implements:

    class MyCapability:
        SUBSCRIPTIONS = (EventKind.RAW_PERCEPT, ...)

        def handle(self, event: BrainEvent, bus: EventBus) -> None:
            # process event; emit responses via bus.publish(...)
            ...

The bus calls `handle` for every event matching SUBSCRIPTIONS.
Subscribers should be idempotent and side-effect-explicit
(emit events; don't mutate other capabilities' state).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .events import BrainEvent, EventKind


logger = logging.getLogger(__name__)


# A subscriber is anything callable as `subscriber.handle(event, bus)`
# OR just `subscriber(event, bus)` — supporting both lets simple
# function-based subscribers exist alongside class-based capabilities.
Subscriber = Any


class EventBus:
    """In-process synchronous event dispatcher (Phase 1).

    Capabilities subscribe via `subscribe(kinds, subscriber)` and
    receive every event whose `kind` matches the subscription.

    Order semantics: subscribers are notified in subscription
    order.  Each `publish` is processed synchronously by all
    matching subscribers before publish returns.

    Re-entrancy: a subscriber MAY publish new events during
    `handle`; those are queued and dispatched after the current
    event completes.  Prevents stack blow-up on
    publish→handle→publish chains.
    """

    def __init__(self) -> None:
        # kind → list of subscribers (preserves registration order)
        self._subscribers: Dict[
            EventKind, List[Subscriber]] = defaultdict(list)
        # Wildcard subscribers see EVERY event (used by loggers/tests).
        self._wildcard: List[Subscriber] = []
        # Re-entrant publish queue.
        self._queue: List[BrainEvent] = []
        self._dispatching: bool = False
        # Counters for diagnostics.
        self.published_total: int = 0
        self.delivered_total: int = 0
        self.errors_caught: int = 0

    # ---- subscription ----

    def subscribe(self,
                    kinds: Tuple[EventKind, ...],
                    subscriber: Subscriber) -> None:
        """Register `subscriber` for the given event kinds."""
        for k in kinds:
            self._subscribers[k].append(subscriber)

    def subscribe_all(self, subscriber: Subscriber) -> None:
        """Register `subscriber` for ALL events (wildcard)."""
        self._wildcard.append(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Remove `subscriber` from all subscription lists.
        Idempotent."""
        for lst in self._subscribers.values():
            while subscriber in lst:
                lst.remove(subscriber)
        while subscriber in self._wildcard:
            self._wildcard.remove(subscriber)

    # ---- publish + dispatch ----

    def publish(self, event: BrainEvent) -> None:
        """Publish an event.  Dispatched synchronously (Phase 1).
        If we're already inside a dispatch (re-entrant publish),
        queue and process after the current event completes."""
        self.published_total += 1
        if self._dispatching:
            self._queue.append(event)
            return
        self._dispatch(event)
        # Drain any queued re-entrant events.
        while self._queue:
            nxt = self._queue.pop(0)
            self._dispatch(nxt)

    def _dispatch(self, event: BrainEvent) -> None:
        self._dispatching = True
        try:
            # Specific subscribers first, then wildcard.
            for sub in list(self._subscribers.get(event.kind, [])):
                self._deliver(sub, event)
            for sub in list(self._wildcard):
                self._deliver(sub, event)
        finally:
            self._dispatching = False

    def _deliver(self,
                   subscriber: Subscriber,
                   event: BrainEvent) -> None:
        try:
            handler = getattr(subscriber, 'handle', None)
            if callable(handler):
                handler(event, self)
            elif callable(subscriber):
                subscriber(event, self)
            else:
                # Not a valid subscriber — silently skip.
                return
            self.delivered_total += 1
        except Exception:
            # One subscriber's failure must not break the bus,
            # but it MUST be visible — silent degradation is the
            # anti-pattern that hides subscriber bugs.  Log + count.
            logger.exception(
                "Subscriber %r failed handling %s",
                subscriber, event.kind)
            self.errors_caught += 1

    # ---- introspection ----

    def stats(self) -> Dict[str, int]:
        return {
            'published_total': self.published_total,
            'delivered_total': self.delivered_total,
            'errors_caught': self.errors_caught,
            'wildcard_subscribers': len(self._wildcard),
            'kind_subscribers': sum(
                len(v) for v in self._subscribers.values()),
        }
