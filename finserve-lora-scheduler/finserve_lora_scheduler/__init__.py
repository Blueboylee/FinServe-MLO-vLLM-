"""
FinServe LoRA Scheduler — vLLM Plugin

通过 vLLM 的 general_plugins 入口点自动加载。
在所有 vLLM 进程（主进程 / EngineCore / Worker）启动前执行 patch，
确保 Scheduler 在实例化之前就已被修改。

用法:
  pip install -e /path/to/finserve-lora-scheduler
  # 然后正常启动 vLLM，plugin 自动生效

环境变量:
  FINSERVE_LORA_REORDER=1       启用/禁用 LoRA 亲和性重排 (默认: 1)
  FINSERVE_LORA_MAX_WAIT_SEC=10 反饥饿阈值，超过此时间的请求强制置顶 (默认: 10)
  FINSERVE_LORA_GROUP_CAP=0     同一 adapter 连续调度上限，0=不限 (默认: 0)
"""

import logging

logger = logging.getLogger("finserve.lora_scheduler")


def register() -> None:
    """vLLM general_plugins 入口点。

    vLLM 在每个进程（包括 EngineCore 子进程）的最早期阶段调用
    load_general_plugins()，该函数会发现并执行此入口点。

    时序保证：
      load_general_plugins() → register() → Scheduler 类定义被使用
    所以我们在这里 patch Scheduler 类是安全的。
    """
    logger.info("=" * 60)
    logger.info("  FinServe LoRA-Aware Scheduler Plugin Initializing")
    logger.info("=" * 60)

    from finserve_lora_scheduler.patches.lora_grouped_prefill import (
        apply_lora_grouped_prefill_patch,
    )

    apply_lora_grouped_prefill_patch()

    logger.info("  FinServe LoRA Scheduler Plugin Ready")
    logger.info("=" * 60)
