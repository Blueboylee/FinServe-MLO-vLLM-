# 优化合理性分析

从「优化动机、实现假设、实测结果」三方面重新审视本项目的各项优化是否合理。

---

## 一、Fused SGMV（Shrink + Expand 寄存器融合）

### 1.1 优化动机（合理）

- 非融合：shrink 写回 `y[total_tokens, rank]`，expand 再读入，产生 **2 × total_tokens × rank × 2B** 的中间层 DRAM 流量 + 一次 kernel 启动。
- 融合：中间 `y[rank]` 驻留寄存器，不写 DRAM，理论上可省上述流量并少一次 launch。方向正确。

### 1.2 实现假设（不成立）

代码注释（sgmv_fused.py 第 55–57 行）写的是：

> 注意: Phase-1 在 grid 的每个 (token, n_tile) 处重复执行。  
> 当 rank 很小 (<=64) 时, **Phase-1 开销远小于 Phase-2**, 重复执行的代价 < 额外 kernel launch + DRAM 往返。

当前 grid 为：

```text
grid = (total_tokens, cdiv(output_dim, BLOCK_N))
```

即每个 program 对应一个 **(token, output_tile)**。每个 program 内都会执行：

1. **Phase-1 (Shrink)**：`x[tid] @ A → intermediate[R]`（整次 shrink）
2. **Phase-2 (Expand)**：`intermediate @ B[:, 当前 tile] → delta[BLOCK_N]`

因此对**同一个 token**，shrink 会在 **num_tiles = cdiv(output_dim, BLOCK_N)** 个 program 里各做一遍，即 **shrink 被重复执行了 num_tiles 次**。

以典型参数估算（hidden_dim=4096, output_dim=4096, rank=64, BLOCK_N=256）：

- num_tiles = 4096 / 256 = **16**
- 每个 program 内：
  - Phase-1：x @ A → 4096×64 ≈ **262K FLOPs**
  - Phase-2：intermediate @ B_tile → 64×256 ≈ **16K FLOPs**
- 即 **Phase-1 单次就比 Phase-2 大约 16 倍**，与「Phase-1 开销远小于 Phase-2」的假设相反。
- 再乘上「每个 token 做 16 次 shrink」：每 token 的 shrink 总工作量是最优的 **16 倍**。

也就是说：**用 16 倍于必要的 shrink 计算，去换掉 1 次中间层写回 + 1 次 kernel launch**，从算力/带宽比上看非常不划算，这是**算法/设计层面的问题**，不是调参能解决的。

### 1.3 结论：Fused SGMV 当前形态不合理

- **动机**：省中间层 DRAM 与 launch，合理。
- **实现**：grid 设计导致「每 token 重复做 num_tiles 次 shrink」，与「Phase-1 远小于 Phase-2」的假设不符，**实现不合理**。
- **合理做法**：应保证 **shrink 每 token 只算一次**，例如：
  - grid 设为 `(total_tokens,)`，每个 program 负责一个 token，在寄存器里算一次 shrink，再在**同一 program 内**用循环遍历所有 output tile 做 expand；或
  - 用两段 kernel，但通过 shared memory 等把中间结果限制在片上，避免全局 DRAM 往返。

当前实现下实测 SGMV 路径比 baseline 慢约 60–70%（例如 BS=64：1641 vs 507 tok/s），与上述 16x 冗余计算量是同一量级的问题。

---

## 二、FP8 KV Cache（E4M3 + per-head 动态 scale）

### 2.1 优化动机（合理）

- KV cache 从 FP16 压到 FP8，显存约减半，高并发时能减轻 memory bandwidth 压力或提高单机可承载的序列长度/ batch。
- Per-head 动态 scale 比 per-tensor 更贴合 attention 的数值范围，精度风险可控。

### 2.2 实现与假设（合理）

- 在线量化：per-head per-token absmax → scale → E4M3，流程标准。
- 融合反量化 + Q@K^T：避免先把整块 K 反量化成 FP16 再算 score，省显存也省一次全局读，设计合理。
- Ada (CC 8.9) 有原生 FP8 支持，量化/反量化代价小。

### 2.3 实测与结论（合理且一致）

- BS=64：1641.3 → 1767.8 tok/s，**+7.7%**；QPS 8.97 → 9.07，与「高并发 memory-bound、FP8 减带宽」一致。
- BS=1：80.7 vs 80.5 tok/s，几乎无差异，与「低并发 compute-bound、FP8 收益有限」一致。

**结论：FP8 KV Cache 的优化方向、实现和实测都合理，可以视为有效、可落地的优化。**

---

## 三、LoRA + Residual + RMSNorm 三路融合

### 3.1 动机与实现（合理）

- 把 base+delta、residual add、RMSNorm 从多 kernel 多 pass 压成单 kernel 单 pass，减少全局内存读写与 launch，是常见的 fusion 思路，合理。
- 为兼容 torch.compile 改成纯 PyTorch ops + 2D/3D 分支，实现上也是合理取舍。

### 3.2 实测（无法单独评估）

- 当前与 SGMV fused 路径绑在一起，没有「仅开 RMSNorm 融合、不开 SGMV」的对照，因此无法从现有 benchmark 中单独判断该融合的收益。
- 若在「正确实现的 SGMV 或原生 LoRA 路径」上单独开启，理论上应有小幅带宽与延迟收益；需后续做隔离实验验证。

**结论：设计合理，但缺少独立对照，无法从现有数据判断实际收益。**

---

## 四、集成层与工程因素（SGMV 变慢的额外原因）

除上述 grid 导致的冗余计算外，集成方式也放大了 SGMV 的劣势：

1. **Per-slice 调用 + contiguous/copy**  
   vLLM 的 QKV 投影有多个 output_slices，当前对每个 slice 单独调一次 `fused_sgmv`，且对列 slice 做 `.contiguous()` 和 `copy_()` 回写，带来额外拷贝与多次 kernel 启动，而 vLLM 原生路径可能是更少的、批量化调用。

2. **Triton autotune**  
   新 shape 会触发 autotune，在多轮、多 batch size 的 benchmark 里会混入 warmup 成本；但相比 16x 的冗余 shrink，这是次要因素。

这些都不改变「Fused SGMV 当前 grid 设计不合理」这一核心结论，但说明：即便修好 grid，也需要在集成层做「少调用、少拷贝」的优化，才能和 vLLM 原生 LoRA 路径公平对比。

---

## 五、总结表

| 优化项 | 动机是否合理 | 实现/假设是否合理 | 实测是否支持 | 总体结论 |
|--------|--------------|-------------------|--------------|----------|
| Fused SGMV | 是 | **否**（grid 导致 shrink 重复 num_tiles 次，且「Phase-1 远小于 Phase-2」不成立） | 否（明显变慢） | **当前形态不合理**，需改 grid 或等价地「每 token 只算一次 shrink」 |
| FP8 KV Cache | 是 | 是 | 是（+7.7% @ BS=64，低并发无收益） | **合理且有效** |
| RMSNorm 三路融合 | 是 | 是 | 无法单独判断 | **设计合理，需单独实验验证** |
| CUDA Graph / custom_op | 是 | 是 | 未在 benchmark 中单独测 | **设计合理，工程上为兼容性所需** |

---

## 六、建议

1. **Fused SGMV**  
   - 必须改：保证每个 token 的 shrink 只算一次（例如单 token、多 tile 的 grid，或等价结构），再重新测；否则该优化在算法层面不成立。
2. **FP8 KV Cache**  
   - 可保留并作为主要可量化的优化点（+7.7% @ BS=64，数据真实）。
3. **RMSNorm 融合**  
   - 在「未启用有问题的 SGMV 路径」下单独做 A/B 测试，再决定是否作为卖点。
4. **对外表述**  
   - 建议明确区分：FP8 KV 为实测有效；SGMV 融合为「方向正确、当前实现存在重复计算问题、待修正后复测」。
