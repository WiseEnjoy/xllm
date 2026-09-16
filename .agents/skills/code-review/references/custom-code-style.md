# Custom Code Style

Project-specific coding style for xllm. The reviewer **MUST** enforce these style.

---

## 1. Naming Conventions

### C++

| Element          | Style                              | Example                              |
|------------------|------------------------------------|--------------------------------------|
| Namespace        | `snake_case`                       | `xllm`, `xllm::detail`              |
| Class / Struct   | `PascalCase`                       | `LlmModelImplBase`, `KVCache`       |
| Function         | `snake_case`                       | `get_input_embeddings`, `forward`    |
| Member variable  | `snake_case_` (trailing underscore)| `model_type_`, `embed_tokens_`       |
| Local variable   | `snake_case`                       | `inputs_embeds`, `kv_caches`         |
| Constant         | `k` + `PascalCase`                | `kContentLength`, `kMaxBatchSize`    |
| Enum type        | `PascalCase`                       | `EngineType`, `DeviceType`           |
| Enum value       | `ALL_CAPS`                         | `LLM`, `VLM`, `INVALID`             |
| Template param   | `PascalCase`                       | `DecoderLayerType`                   |
| Macro            | `ALL_CAPS`                         | `XLLM_CHECK`, `LOG_EVERY_N`         |
| File name        | `snake_case`                       | `llm_model_base.h`, `types.h`       |
| Header guard     | `#pragma once`                     | -                                    |

### Python

| Element          | Style                  | Example                              |
|------------------|------------------------|--------------------------------------|
| Module / file    | `snake_case`           | `model_loader.py`                    |
| Class            | `PascalCase`           | `TokenizerConfig`                    |
| Function         | `snake_case`           | `load_model`                         |
| Variable         | `snake_case`           | `batch_size`                         |
| Constant         | `ALL_CAPS`             | `MAX_SEQ_LEN`                        |
| Private member   | `_leading_underscore`  | `_internal_state`                    |

---

## 2. File & Header Rules

- **Copyright header required** on all new files. Use the correct year matching the file creation date.

```cpp
/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
...
==============================================================================*/
```

- **No relative paths in `#include`**. Always use project-root-relative paths.

```cpp
// Good
#include "core/common/types.h"

// Bad
#include "../common/types.h"
#include "./types.h"
```

- **Remove redundant and duplicate includes**. Each header should be included exactly once, and unused includes must be cleaned up.

---

## 3. Type System & Declarations

- **Use fixed-width integers** (`int32_t`, `int64_t`) instead of plain `int`, unless the API you are calling explicitly requires `int`.

```cpp
// Good
int32_t batch_size = 16;
int64_t total_tokens = 0;

// Bad
int batch_size = 16;
```

- **Use `static_cast`** for all type conversions. Never use C-style casts.

```cpp
// Good
auto len = static_cast<int32_t>(vec.size());

// Bad
auto len = (int32_t)vec.size();
```

- **Do not use `auto` for simple/primitive types**. `auto` is acceptable for complex types (iterators, lambdas, template-deduced types) but not for `int32_t`, `float`, `bool`, `std::string`, etc.

```cpp
// Good
int32_t count = 0;
auto it = map.find(key);  // complex iterator type, auto is fine

// Bad
auto count = 0;
auto name = std::string("model");
```

- **Use `using` instead of `typedef`** for type aliases. Prefer aliases for complex types to improve readability.

```cpp
// Good
using TensorVec = std::vector<torch::Tensor>;
using CallbackFn = std::function<void(int32_t)>;

// Bad
typedef std::vector<torch::Tensor> TensorVec;
```

- **Use `enum class`** instead of plain `enum` to provide type safety and prevent implicit conversions.

```cpp
// Good
enum class DeviceType : int8_t { CPU = 0, CUDA = 1, NPU = 2 };

// Bad
enum DeviceType { CPU = 0, CUDA = 1, NPU = 2 };
```

- **Use `nullptr`** instead of `NULL` or `0` for null pointers.

- **Choose the right container**: use `std::unordered_map` / `std::unordered_set` when key ordering is irrelevant (O(1) average lookup). Use `std::map` / `std::set` only when sorted iteration or key ordering is required.

---

## 4. Class Design

- **Mark classes `final`** if they are not designed to be inherited from.

```cpp
// Good
class TokenizerConfig final { ... };

// Bad – class has no virtual functions and is not intended as a base
class TokenizerConfig { ... };
```

- **Use `explicit`** on any constructor that can be invoked with a single argument. This includes multi-parameter constructors where all parameters except the first have default values.

```cpp
// Good
explicit ModelArgs(const std::string& path, int32_t num_layers = 12);

// Bad – allows implicit conversion from std::string
ModelArgs(const std::string& path, int32_t num_layers = 12);
```

- **Use `override`** when overriding virtual functions in derived classes. Never repeat the `virtual` keyword on overrides.

```cpp
// Good
ModelOutput forward(torch::Tensor tokens, ...) override;

// Bad – missing override
ModelOutput forward(torch::Tensor tokens, ...);

// Bad – redundant virtual on override
virtual ModelOutput forward(torch::Tensor tokens, ...) override;

// Bad – missing override (virtual-only in derived class)
virtual ModelOutput forward(torch::Tensor tokens, ...);
```

- **Structs must not have member functions**. If you need methods, use a `class`. Structs are for plain data aggregation only.

---

## 5. Memory & Resource Management

- **Avoid raw pointers**. Prefer smart pointers (`std::unique_ptr`, `std::shared_ptr`) for ownership semantics.
  - Use `std::unique_ptr` by default (sole ownership).
  - Use `std::shared_ptr` only when shared ownership is genuinely needed.
  - Raw pointers are acceptable only for non-owning references where the lifetime is clearly managed elsewhere.

---

## 6. Scoping & Visibility

### C++

- **File-local functions and variables** (used only within a single `.cpp` file) must be placed in an **anonymous namespace**.

```cpp
namespace {
int32_t compute_padding(int32_t seq_len, int32_t alignment) {
  return (alignment - seq_len % alignment) % alignment;
}
}  // namespace
```

### Python

- **File-local helper functions** (not part of the public API) must be prefixed with `_`.
- **Non-public member functions** of a class must be prefixed with `_`.

```python
def _validate_config(config: dict) -> bool:
    ...

class ModelLoader:
    def load(self, path: str) -> Model:
        self._check_path(path)
        ...

    def _check_path(self, path: str) -> None:
        ...
```

---

## 7. Torch & Framework API Usage

- **Use `torch::` namespace** instead of `at::` or `c10::` wherever possible. Prefer the highest-level PyTorch C++ API.

```cpp
// Good
torch::Tensor output = torch::zeros({batch_size, hidden_dim});

// Bad
at::Tensor output = at::zeros({batch_size, hidden_dim});
c10::optional<torch::Tensor> mask = c10::nullopt;  // use std::optional
```

- **Use `CHECK`** (glog) instead of `TORCH_CHECK` for assertions.

```cpp
// Good
CHECK(tensor.is_contiguous()) << "Input tensor must be contiguous";

// Bad
TORCH_CHECK(tensor.is_contiguous(), "Input tensor must be contiguous");
```

- **Use `LOG(FATAL)`** for unrecoverable errors instead of throwing `std::runtime_error`.

```cpp
// Good
LOG(FATAL) << "Unsupported model type: " << model_type;

// Bad
throw std::runtime_error("Unsupported model type: " + model_type);
```

---

## 8. Code Style & Control Flow

- **Always use braces `{}`** with `if`, `while`, `for`, even for single-line bodies.

```cpp
// Good
if (x > 0) {
  return x;
}

// Bad
if (x > 0) return x;
```

- **Avoid `if` inside `for` loops** when possible. Prefer filtering the data beforehand or restructuring the logic (e.g., early `continue`, separate loops, `std::copy_if`).

- **Define variables close to first use**. Do not declare all variables at the top of a function.

- **Annotate constant arguments** with a comment indicating the parameter name when calling functions or constructors.

```cpp
// Good
auto layer = DecoderLayer(/*hidden_size=*/4096, /*num_heads=*/32);

// Bad
auto layer = DecoderLayer(4096, 32);
```

- **Prefer lambdas over `std::bind`**. Lambdas have explicit capture lists and parameter types. When a parameter is unused, annotate it with `/*unused*/`.

```cpp
// Good
auto fn = [this](const etcd::Response& response, uint64_t prefix_len) {
  handle_watch(response, prefix_len);
};
[](void* /*unused*/) { request_in_metric(nullptr); }

// Bad
auto fn = std::bind(&MyClass::handle_watch, this,
                    std::placeholders::_1, std::placeholders::_2);
std::bind(request_in_metric, nullptr)
```

---

## 9. STL Best Practices

- **Always `reserve()` before filling a `std::vector`** when the size is known or can be estimated.

```cpp
// Good
std::vector<torch::Tensor> outputs;
outputs.reserve(num_layers);
for (int32_t i = 0; i < num_layers; ++i) {
  outputs.emplace_back(compute_layer(i));
}

// Bad – causes multiple reallocations
std::vector<torch::Tensor> outputs;
for (int32_t i = 0; i < num_layers; ++i) {
  outputs.push_back(compute_layer(i));
}
```

- **Prefer `emplace_back`** over `push_back` to construct elements in-place and avoid unnecessary copies.

---

## 10. Global Flags

- **Do not use `FLAGS_` global variables**. Prefer passing configuration through constructor parameters or config structs.
- **Register new flags in their config category**. When adding a flag under `xllm/core/framework/config/`, add it to that config class's `option_category()` so `core/framework/config/help_formatter.h` can print it in `--help`.
- **Keep config defaults explicit**. Primitive `PROPERTY` fields such as `bool`, `int32_t`, `int64_t`, and `double` must keep an in-class default matching the corresponding `DEFINE_*` default.
---

## 11. Python-Specific Rules

- **Type annotations are required** on all function signatures (parameters and return types). Use `typing` module types where needed.

```python
# Good
def load_model(path: str, device: str = "cuda") -> nn.Module:
    ...

# Bad
def load_model(path, device="cuda"):
    ...
```

- **Private helpers**: prefix with `_` (see Section 6).
- **Prefer absolute imports** rooted at the top-level package for project code.
  - A single-dot relative import (`from .module import Name`) is acceptable for a sibling module in the same package, especially for package-local implementation details and `__init__.py` re-exports.
  - Avoid parent-relative imports (`from ..module import Name`) and deeper relative imports. Use an absolute import from `xllm` instead so the dependency remains clear when code moves between packages.

```python
# Preferred across packages
from xllm.python.attention.backend import AttentionBackend

# Acceptable within the same package
from .backend import AttentionBackend

# Avoid
from ..attention.backend import AttentionBackend
```

---

## 12. Python Logging

All Python diagnostic output **MUST** go through the shared logger at `scripts/logger.py`. Do not use `print()` for logging, do not call `logging.basicConfig`, and do not create per-module loggers via `logging.getLogger(...)`.

- **Import:** `from scripts.logger import logger`. For standalone `tools/*.py`, prepend repo root to `sys.path` first: `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`.
- **Levels:** `debug` (investigation only), `info` (normal progress), `warning` (recoverable anomaly / fallback), `error` (failure, including pre-`exit(1)`), `fatal` (process-fatal). Inside `except`, prefer `logger.exception(...)` to keep the traceback.
- **No level prefixes in messages.** The formatter already emits `I/W/E/F/D` plus `file:line`. Status emojis (`✅ ❌ ⚠️ 🔨 🚀 ℹ️`) are allowed.
- **`print()` is only for program output**, not diagnostics: model results in `examples/`, CLI stdout meant to be piped, or verbatim streaming of subprocess output (e.g. `sys.stdout.write(line)` for ctest).
- **Do not reconfigure the shared logger** from library code: no `setLevel`, no extra handlers, no `propagate` changes.

```python
# Good
from scripts.logger import logger
logger.info(f"compiled kernel: {name}")
logger.warning("Skip reference check: NPU is not available")
try:
    do_work(path)
except OSError:
    logger.exception(f"failed to process {path}")
```

```python
# Bad
import logging
logger = logging.getLogger(__name__)            # re-implementing a logger
print(f"[INFO] build start: device={device}")   # diagnostic via print
logger.warning("[WARN] missing --device")       # duplicated level prefix
logger.info("build failed, exiting")            # wrong level for a failure
```

## 13. Shell 操作纪律（执行环境规则）

- **`pgrep`/`ps | grep` 会匹配到自身所在的 shell 命令行**（bash -c 的 argv 里含关键词），导致误判“进程仍在运行”。判断后台进程存活必须用 `pgrep -f "<pattern>" | grep -v $$` 或先拿 PID 再 `ps -p <PID>`，禁止直接 `pgrep -f <脚本名>` 后当作存活证据。
- **后台长任务必须落盘日志 + 短轮询**：`nohup ... > log 2>&1 &` 后立即返回，用固定间隔（如 15-30s）轮询日志/端口，绝不 sleep 挂死单条命令直到工具超时。

## 14. 容器环境的 NPU 限制（执行环境规则）

- **`npu-smi set -t reset` 在容器内不可用**（报 "This command cannot be executed on a container"）。NPU 级重置（reset/功率/时钟等 set 类操作）只能由宿主机执行，容器内只能查询（`npu-smi info`）。
- **HCCL channel 资源泄漏的容器内恢复手段有限**：`Acquire HCCL channel failed, ret: 4` / `HcclCreateOpResCtx failed` 通常由 kill -9 的进程残留导致。容器内只能：①确保杀干净相关进程（含 forkserver 子进程）+ 等待设备回收（观察 `npu-smi info` HBM 回落）；②清理 `/dev/shm` 非 xllm 残留与 torch_extensions JIT 缓存；③泄漏不恢复时，唯一彻底手段是**请求宿主机管理员重置对应卡**，或重启容器。
- 推论：**不要 kill -9 带 HCCL 通信的进程**（会造成上述泄漏）；测试脚本要处理 SIGTERM 优雅退出，或用 `timeout` 包裹让进程自然结束。
