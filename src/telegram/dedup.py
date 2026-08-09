from collections import OrderedDict


class UpdateDeduplicator:
    """Bounded memory of processed update ids.

    Telegram re-sends an update whenever the webhook answers slowly or
    non-200; without this, every retry would run the whole LLM pipeline
    again and the user would get duplicate replies.
    """

    def __init__(self, *, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._seen: OrderedDict[int, None] = OrderedDict()
        self._capacity = capacity

    def seen_before(self, update_id: int) -> bool:
        """True when the id was already registered; registers it otherwise."""
        if update_id in self._seen:
            self._seen.move_to_end(update_id)
            return True
        self._seen[update_id] = None
        while len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
        return False
