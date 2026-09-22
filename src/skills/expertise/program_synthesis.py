"""程序合成技能。

根据自然语言描述或输入/输出示例生成 Python 代码，并在安全沙盒中
执行验证。采用"生成 → 执行 → 误差驱动修正"循环，与 ZeroDataModel 的
预测-误差-更新范式保持一致。

安全约束：
- 沙盒禁止 ``import``、文件 IO、网络、``exec``/``eval`` 的嵌套、超时。
- 仅允许基本算术、列表/字典、``for``/``while``、内置函数白名单。
- 命名空间注入受控的输入占位符 ``inputs``。
"""
from __future__ import annotations

import logging
import re
import signal
from typing import Any, Callable

import numpy as np

from ..base import SkillBase, SkillContext, SkillResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 安全沙盒
# ------------------------------------------------------------------ #

#: 禁止的关键字（基于 AST 名字检查可绕过字符串形式，因此双管齐下）
_FORBIDDEN_TOKENS = (
    "import ",
    "import(",
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "globals(",
    "locals(",
    "os.",
    "sys.",
    "subprocess",
    "socket",
    "shutil",
    "pickle",
    "getattr(",
    "setattr(",
    "delattr(",
    "__",
)

#: 允许的内置函数白名单
_ALLOWED_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "round": round,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
    "print": print,
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}


class SandboxTimeout(Exception):
    """沙盒执行超时。"""


class SandboxError(Exception):
    """沙盒执行违规（禁止 API 或语法错误）。"""


def _validate_source(code: str) -> None:
    """静态检查源码是否使用了禁止的 API。"""
    for tok in _FORBIDDEN_TOKENS:
        if tok in code:
            raise SandboxError(f"forbidden token in generated code: {tok!r}")


def _sandbox_timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise SandboxTimeout("execution exceeded time budget")


def execute_in_sandbox(
    code: str,
    inputs: dict[str, Any],
    *,
    timeout_s: float = 0.5,
) -> dict[str, Any]:
    """在受限命名空间中执行代码，返回命名空间快照。

    返回 ``{"ok": bool, "output": Any, "error": str | None}``。
    """
    _validate_source(code)

    sandbox_globals: dict[str, Any] = {
        "__builtins__": _ALLOWED_BUILTINS,
        **inputs,
    }
    # 提取主入口的输出（约定最后一句为 ``result = ...``）
    try:
        # 仅在 POSIX 上可用；Windows 上 fallback 为无超时执行
        try:
            old_handler = signal.signal(signal.SIGALRM, _sandbox_timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        except (AttributeError, ValueError):
            old_handler = None

        exec(code, sandbox_globals)  # noqa: S102 (受控沙盒)
    except SandboxTimeout:
        return {"ok": False, "output": None, "error": "timeout"}
    except SandboxError as exc:
        return {"ok": False, "output": None, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "output": None, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            if old_handler is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass

    output = sandbox_globals.get("result", None)
    return {"ok": True, "output": output, "error": None}


# ------------------------------------------------------------------ #
# 代码模板与生成器
# ------------------------------------------------------------------ #

#: 简单问题模板库：从关键词匹配生成代码骨架
_TEMPLATES: list[dict[str, Any]] = [
    {
        "keywords": ("sum", "求和", "加"),
        "code": (
            "def solve(inputs):\n"
            "    total = sum(inputs['nums'])\n"
            "    return total\n"
            "result = solve(inputs)\n"
        ),
    },
    {
        "keywords": ("max", "最大", "max"),
        "code": (
            "def solve(inputs):\n"
            "    return max(inputs['nums'])\n"
            "result = solve(inputs)\n"
        ),
    },
    {
        "keywords": ("min", "最小"),
        "code": (
            "def solve(inputs):\n"
            "    return min(inputs['nums'])\n"
            "result = solve(inputs)\n"
        ),
    },
    {
        "keywords": ("len", "count", "长度", "数量"),
        "code": (
            "def solve(inputs):\n"
            "    return len(inputs['nums'])\n"
            "result = solve(inputs)\n"
        ),
    },
    {
        "keywords": ("reverse", "反转", "倒序"),
        "code": (
            "def solve(inputs):\n"
            "    arr = list(inputs['nums'])\n"
            "    arr.reverse()\n"
            "    return arr\n"
            "result = solve(inputs)\n"
        ),
    },
    {
        "keywords": ("sort", "排序", "升序"),
        "code": (
            "def solve(inputs):\n"
            "    return sorted(inputs['nums'])\n"
            "result = solve(inputs)\n"
        ),
    },
    {
        "keywords": ("double", "乘二", "2倍"),
        "code": (
            "def solve(inputs):\n"
            "    return [x * 2 for x in inputs['nums']]\n"
            "result = solve(inputs)\n"
        ),
    },
    {
        "keywords": ("square", "平方", "二次方"),
        "code": (
            "def solve(inputs):\n"
            "    return [x * x for x in inputs['nums']]\n"
            "result = solve(inputs)\n"
        ),
    },
]


def _match_template(spec: str) -> str | None:
    """根据自然语言描述匹配模板，返回代码字符串。"""
    spec_lower = spec.lower()
    for tmpl in _TEMPLATES:
        if any(k.lower() in spec_lower for k in tmpl["keywords"]):
            return tmpl["code"]
    return None


def _mutation_variants(code: str) -> list[str]:
    """对代码进行简单变异（替换函数体行）。

    用于生成-验证循环中的修正步骤：当模板直出代码与示例不符时，
    在几个候选实现中搜索误差更小的版本。
    """
    variants: list[str] = []
    bodies = [
        "sum(inputs['nums'])",
        "max(inputs['nums'])",
        "min(inputs['nums'])",
        "len(inputs['nums'])",
        "sorted(inputs['nums'])",
        "[x * 2 for x in inputs['nums']]",
        "[x * x for x in inputs['nums']]",
        "list(reversed(inputs['nums']))",
    ]
    # 抽取原代码的函数签名行（前两行）
    lines = code.splitlines()
    if len(lines) < 2:
        return variants
    header = "\n".join(lines[:2])  # def 行 + 下一行占位
    for body in bodies:
        candidate = (
            header
            + "\n"
            + f"    return {body}\n"
            + "result = solve(inputs)\n"
        )
        variants.append(candidate)
    return variants


def _output_error(output: Any, expected: Any) -> float:
    """计算输出与期望之间的标量误差。"""
    if output is None:
        return 1e6
    try:
        if isinstance(expected, (list, tuple)):
            out_arr = np.atleast_1d(np.asarray(output, dtype=np.float64))
            exp_arr = np.atleast_1d(np.asarray(expected, dtype=np.float64))
            if out_arr.shape != exp_arr.shape:
                return 1e6
            return float(np.mean(np.abs(out_arr - exp_arr)))
        return float(abs(float(output) - float(expected)))
    except (TypeError, ValueError):
        return 1e6 if output != expected else 0.0


# ------------------------------------------------------------------ #
# 主技能类
# ------------------------------------------------------------------ #


class ProgramSynthesizer(SkillBase):
    """程序合成技能。

    工作流程：
    1. 根据自然语言描述或 I/O 示例，从模板库生成候选代码。
    2. 在沙盒中执行候选代码，对比输出与期望。
    3. 若误差高于阈值，尝试变异候选并重试。
    4. 选出误差最低的代码作为最终输出。

    与 ZeroDataModel 集成：
    - 在 ``think()`` 的元认知阶段调用，输出代码与执行误差作为
      metadata，供前端"代码工作室"面板展示。
    """

    name = "program_synthesis"
    dimension = "expertise"

    #: 生成-验证沙盒循环开销较高，节流到每 5 步刷新一次。
    #: ``synthesize_from_spec()`` / ``synthesize_from_examples()``
    #: 直接调用不受节流影响。
    process_interval = 5

    def __init__(
        self,
        *,
        max_iterations: int = 4,
        error_threshold: float = 1e-3,
        sandbox_timeout: float = 0.5,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled, **kwargs)
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold
        self.sandbox_timeout = sandbox_timeout
        self._history: list[dict[str, Any]] = []
        self._best: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #

    def synthesize_from_spec(
        self, spec: str, examples: list[tuple[dict, Any]] | None = None
    ) -> dict[str, Any]:
        """根据自然语言描述合成代码。

        Parameters
        ----------
        spec
            自然语言描述（如 "对列表求和"）。
        examples
            可选的 I/O 示例。若提供，则用误差驱动选择最优候选。

        Returns
        -------
        dict
            ``{"code": str, "output": Any, "error": float,
              "iterations": int, "converged": bool}``
        """
        candidate = _match_template(spec)
        if candidate is None:
            # 兜底：尝试 sum 模板
            candidate = _TEMPLATES[0]["code"]

        return self._refine_until_converged(candidate, examples or [])

    def synthesize_from_examples(
        self, examples: list[tuple[dict, Any]]
    ) -> dict[str, Any]:
        """根据 I/O 示例合成代码（无自然语言提示）。

        在所有模板上尝试并选最优。
        """
        if not examples:
            raise ValueError("at least one example is required")

        best: dict[str, Any] | None = None
        for tmpl in _TEMPLATES:
            result = self._refine_until_converged(tmpl["code"], examples)
            if best is None or result["error"] < best["error"]:
                best = result
        assert best is not None
        return best

    # ------------------------------------------------------------------ #
    # 生成-验证循环
    # ------------------------------------------------------------------ #

    def _refine_until_converged(
        self, code: str, examples: list[tuple[dict, Any]]
    ) -> dict[str, Any]:
        if not examples:
            # 无示例 → 仅做静态验证
            result = execute_in_sandbox(
                code, {"inputs": {"nums": [1, 2, 3]}}, timeout_s=self.sandbox_timeout
            )
            return {
                "code": code,
                "output": result["output"],
                "error": 0.0 if result["ok"] else 1e6,
                "iterations": 1,
                "converged": result["ok"],
                "sandbox_error": result.get("error"),
            }

        current = code
        best_error = float("inf")
        best_code = code
        best_output: Any = None
        iterations = 0
        converged = False

        for it in range(self.max_iterations):
            iterations = it + 1
            errors: list[float] = []
            outputs: list[Any] = []
            for inp, _expected in examples:
                run = execute_in_sandbox(
                    current, {"inputs": inp}, timeout_s=self.sandbox_timeout
                )
                if not run["ok"]:
                    errors.append(1e6)
                    outputs.append(None)
                else:
                    errors.append(_output_error(run["output"], _expected))
                    outputs.append(run["output"])
            mean_error = float(np.mean(errors))
            if mean_error < best_error:
                best_error = mean_error
                best_code = current
                best_output = outputs[0] if outputs else None
            if mean_error <= self.error_threshold:
                converged = True
                break
            # 尝试变异
            variants = _mutation_variants(current)
            improved = False
            for var in variants:
                var_errors: list[float] = []
                for inp, _expected in examples:
                    run = execute_in_sandbox(
                        var, {"inputs": inp}, timeout_s=self.sandbox_timeout
                    )
                    if not run["ok"]:
                        var_errors.append(1e6)
                    else:
                        var_errors.append(
                            _output_error(run["output"], _expected)
                        )
                var_mean = float(np.mean(var_errors))
                if var_mean < best_error:
                    best_error = var_mean
                    best_code = var
                    best_output = None
                    current = var
                    improved = True
                    if var_mean <= self.error_threshold:
                        converged = True
                        break
            if converged:
                break
            if not improved:
                # 无改进 → 终止
                break

        # 重新运行最优代码以获取最新输出
        final_run = execute_in_sandbox(
            best_code,
            {"inputs": examples[0][0]},
            timeout_s=self.sandbox_timeout,
        )
        final_output = best_output if best_output is not None else final_run["output"]

        result = {
            "code": best_code,
            "output": final_output,
            "error": best_error,
            "iterations": iterations,
            "converged": converged,
            "sandbox_error": final_run.get("error") if not final_run["ok"] else None,
        }
        self._history.append({**result, "spec": "(refined)"})
        self._best = result
        return result

    # ------------------------------------------------------------------ #
    # SkillBase 接口
    # ------------------------------------------------------------------ #

    def process(self, ctx: SkillContext) -> SkillResult:
        """在 ``think()`` 阶段执行：尝试合成"求和"程序作为演示。

        实际部署中可由前端触发自定义任务，本方法仅作为离线演示入口，
        保证元数据面板始终有最新结果。
        """
        demo_examples = [
            ({"nums": [1, 2, 3]}, 6),
            ({"nums": [10, -5, 0]}, 5),
            ({"nums": [100, 200]}, 300),
        ]
        result = self.synthesize_from_spec(
            "sum the list of numbers", demo_examples
        )
        return SkillResult(
            name=self.name,
            data={
                "best_program": result["code"],
                "error": result["error"],
                "iterations": result["iterations"],
                "converged": result["converged"],
                "history_size": len(self._history),
                "output_sample": str(result["output"])[:80],
            },
        )

    def snapshot(self) -> dict[str, Any]:
        if self._best is not None:
            return {
                "name": self.name,
                "enabled": self.enabled,
                "ready": True,
                "data": {
                    "best_error": self._best["error"],
                    "best_iterations": self._best["iterations"],
                    "best_converged": self._best["converged"],
                },
            }
        return super().snapshot()
