"""
LoRA-Aware Chunked Prefill Patch for vLLM Scheduler

在每次 schedule() 调用前，对 waiting queue 进行 LoRA 亲和性重排：
  1. 优先调度 LoRA 权重已在 GPU 上的请求（减少 adapter 切换）
  2. 将使用相同 adapter 的请求分组（利用 S-LoRA/Punica BGMV 批量计算）
  3. 反饥饿：等待超过阈值的请求无条件提升优先级

适用场景：Multi-LoRA serving，如 Expert-A / Expert-B 共享基座模型。
"""

import logging
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler

logger = logging.getLogger("finserve.lora_scheduler")

# ---------------------------------------------------------------------------
# 可通过环境变量调整的参数
# ---------------------------------------------------------------------------
LORA_MAX_WAIT_SEC = float(os.environ.get("FINSERVE_LORA_MAX_WAIT_SEC", "10"))
LORA_GROUP_CAP = int(os.environ.get("FINSERVE_LORA_GROUP_CAP", "0"))
LORA_REORDER_ENABLED = os.environ.get(
    "FINSERVE_LORA_REORDER", "1"
) not in ("0", "false", "False", "no")

_reorder_count = 0
_LOG_INTERVAL = 50


def _get_hot_lora_ids(scheduler: "Scheduler") -> set[int]:
    """返回当前 running 请求中正在使用的 LoRA adapter ID 集合。
    这些 adapter 的权重大概率已在 GPU buffer 中，切换成本低。"""
    hot = set()
    for req in scheduler.running:
        lr = getattr(req, "lora_request", None)
        if lr and lr.lora_int_id > 0:
            hot.add(lr.lora_int_id)
    return hot


def _reorder_waiting_queue(scheduler: "Scheduler") -> None:
    """对 waiting queue 执行 LoRA 亲和性重排。

    排序维度（优先级从高到低）：
      0) 饥饿防护 — 等待超过 LORA_MAX_WAIT_SEC 的请求无条件置顶
      1) GPU 亲和 — adapter 已在 GPU 上的请求优先
      2) 分组聚合 — 相同 adapter 的请求连续排列
      3) 原始顺序 — 组内保持 FCFS（用插入序号做稳定排序）
    """
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
            hot = 0  # no-LoRA 请求视为"中性"，不惩罚

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
    """限制同一 LoRA group 连续调度的最大数量，防止单一 adapter 长期霸占 batch。
    超出 cap 的请求被轮转到下一轮。"""
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
    """Monkey-patch Scheduler.schedule() 以注入 LoRA 亲和性重排。

    通过 vLLM general_plugins 在进程启动时调用，确保所有 worker
    进程使用同一个 patched schedule()。
    """
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
        """Wrapper: 在原始 schedule() 前执行 LoRA 亲和性重排。"""
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
