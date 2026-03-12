import logging
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler

logger = logging.getLogger("finserve.lora_scheduler")

LORA_MAX_WAIT_SEC = float(os.environ.get("FINSERVE_LORA_MAX_WAIT_SEC", "10"))
LORA_GROUP_CAP = int(os.environ.get("FINSERVE_LORA_GROUP_CAP", "0"))
LORA_REORDER_ENABLED = os.environ.get(
    "FINSERVE_LORA_REORDER", "1"
) not in ("0", "false", "False", "no")

_reorder_count = 0
_LOG_INTERVAL = 50


def _get_hot_lora_ids(scheduler: "Scheduler") -> set[int]:
    hot = set()
    for req in scheduler.running:
        lr = getattr(req, "lora_request", None)
        if lr and lr.lora_int_id > 0:
            hot.add(lr.lora_int_id)
    return hot


def _reorder_waiting_queue(scheduler: "Scheduler") -> None:
    global _reorder_count

    waiting = scheduler.waiting
    n = len(waiting)
    if n <= 1:
        return

    hot_lora_ids = _get_hot_lora_ids(scheduler)

    requests: list = []
    while waiting:
        requests.append(waiting.pop_request())

    now = time.monotonic()

    def _sort_key(item):
        idx, req = item
        lora_id = 0
        lr = getattr(req, "lora_request", None)
        if lr:
            lora_id = lr.lora_int_id

        wait_time = now - req.arrival_time if hasattr(req, "arrival_time") else 0.0
        starving = 0 if (LORA_MAX_WAIT_SEC > 0 and wait_time > LORA_MAX_WAIT_SEC) else 1

        if lora_id in hot_lora_ids:
            hot = 0
        elif lora_id > 0:
            hot = 1
        else:
            hot = 0

        return (starving, hot, lora_id, idx)

    indexed = list(enumerate(requests))
    indexed.sort(key=_sort_key)

    if LORA_GROUP_CAP > 0:
        indexed = _apply_group_cap(indexed, LORA_GROUP_CAP)

    for _, req in indexed:
        waiting.add_request(req)

    _reorder_count += 1
    if _reorder_count % _LOG_INTERVAL == 1:
        groups = defaultdict(int)
        for _, req in indexed:
            lr = getattr(req, "lora_request", None)
            lid = lr.lora_int_id if lr else 0
            groups[lid] += 1
        logger.info(
            "LoRA reorder #%d: %d requests, hot=%s, groups=%s",
            _reorder_count,
            n,
            hot_lora_ids,
            dict(groups),
        )


def _apply_group_cap(indexed, cap: int):
    from collections import defaultdict

    group_counts: dict[int, int] = defaultdict(int)
    result = []
    overflow = []

    for idx, req in indexed:
        lr = getattr(req, "lora_request", None)
        lora_id = lr.lora_int_id if lr else 0
        if group_counts[lora_id] < cap:
            result.append((idx, req))
            group_counts[lora_id] += 1
        else:
            overflow.append((idx, req))

    result.extend(overflow)
    return result


def apply_lora_grouped_prefill_patch() -> None:
    if not LORA_REORDER_ENABLED:
        logger.info("LoRA grouped prefill is DISABLED (FINSERVE_LORA_REORDER=0)")
        return

    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except ImportError:
        logger.warning(
            "Cannot import vllm.v1.core.sched.scheduler.Scheduler — "
            "LoRA grouped prefill patch will NOT be applied. "
            "Make sure you are using vLLM v0.8+ with V1 engine."
        )
        return

    if hasattr(Scheduler, "_finserve_patched"):
        logger.info("LoRA grouped prefill patch already applied, skipping.")
        return

    original_schedule = Scheduler.schedule

    def lora_aware_schedule(self):
        if getattr(self, "lora_config", None) and self.waiting:
            try:
                _reorder_waiting_queue(self)
            except Exception:
                logger.warning(
                    "LoRA reorder failed, falling back to default order",
                    exc_info=True,
                )
        return original_schedule(self)

    Scheduler.schedule = lora_aware_schedule
    Scheduler._finserve_patched = True

    logger.info(
        "LoRA-Aware Chunked Prefill patch applied successfully. "
        "Config: max_wait=%.1fs, group_cap=%d, enabled=%s",
        LORA_MAX_WAIT_SEC,
        LORA_GROUP_CAP,
        LORA_REORDER_ENABLED,
    )
