"""Discovery Paper Writer — 自动撰写发现"论文"。

当一个假设被验证（接受或拒绝），自动撰写简短 Markdown 论文：
摘要、方法、结果、讨论、结论。存档到 discoveries/ 目录。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from .hypothesis_generator import Hypothesis, _stable_hash_id
from .experiment_designer import ExperimentDesign
from .result_analyzer import AnalysisResult


@dataclass
class Paper:
    """自动撰写的发现论文。"""

    paper_id: str
    title: str
    hypothesis_id: str
    decision: str
    markdown: str
    file_path: str
    timestamp: float


class PaperWriter:
    """论文撰写器。

    Parameters
    ----------
    output_dir : str, default "discoveries"
        论文存档目录。
    """

    def __init__(self, output_dir: str = "discoveries") -> None:
        self.output_dir = output_dir
        self._papers: list[Paper] = []
        self._max_history = 100

    # ------------------------------------------------------------------ #
    # 撰写
    # ------------------------------------------------------------------ #
    def write(
        self,
        hypothesis: Hypothesis,
        design: ExperimentDesign,
        analysis: AnalysisResult,
    ) -> Paper:
        """撰写并存档一篇发现论文。

        Returns
        -------
        Paper
            含完整 markdown 与文件路径。
        """
        timestamp = time.time()
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        decision_cn = {
            "accept": "接受",
            "reject": "拒绝",
            "inconclusive": "不确定",
        }.get(analysis.decision, analysis.decision)

        title = f"发现报告：{hypothesis.statement[:60]}{'...' if len(hypothesis.statement) > 60 else ''}"

        markdown = self._render_markdown(
            hypothesis, design, analysis, title, decision_cn, ts_str
        )

        # 存档
        # H4 修复：os.makedirs 移入 try 块。原代码在 try 外调用
        # makedirs，若目录创建失败（权限不足/只读文件系统/路径冲突）
        # 会直接抛出 OSError 中断 write()，论文无法记录。
        # H4+ 修复：含空字节的非法路径会抛 ValueError（非 OSError
        # 子类），拓宽捕获到 (OSError, ValueError) 以覆盖此类情况。
        # F3 修复：os.path.join 也在 try 外，output_dir 为 None 时
        # 抛 TypeError 不被捕获。将所有文件系统操作移入 try。
        safe_hyp_id = hypothesis.hypothesis_id.replace(" ", "_")
        filename = f"{ts_str}_{safe_hyp_id}_{analysis.decision}.md"
        file_path = ""
        try:
            full_path = os.path.join(self.output_dir, filename)
            os.makedirs(self.output_dir, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            file_path = full_path
        except (OSError, ValueError, TypeError):
            file_path = ""  # 写入失败仍返回内存中的论文

        paper = Paper(
            paper_id=_stable_hash_id(hypothesis.hypothesis_id + ts_str, "P"),
            title=title,
            hypothesis_id=hypothesis.hypothesis_id,
            decision=analysis.decision,
            markdown=markdown,
            file_path=file_path,
            timestamp=timestamp,
        )
        self._papers.append(paper)
        if len(self._papers) > self._max_history:
            self._papers = self._papers[-self._max_history :]
        return paper

    def _render_markdown(
        self,
        h: Hypothesis,
        d: ExperimentDesign,
        a: AnalysisResult,
        title: str,
        decision_cn: str,
        ts_str: str,
    ) -> str:
        """渲染 Markdown 论文。"""
        bf_interpretation = (
            "强证据" if a.bayes_factor > 50
            else "中等证据" if a.bayes_factor > 10
            else "弱证据" if a.bayes_factor > 1
            else "证据不足"
        )

        return f"""# {title}

**论文 ID**: {_stable_hash_id(h.hypothesis_id + ts_str, "P").split("_", 1)[1]}
**假设 ID**: {h.hypothesis_id}
**实验 ID**: {d.experiment_id}
**分析 ID**: {a.analysis_id}
**生成时间**: {ts_str}
**决策**: {decision_cn}

---

## 摘要

本研究测试了以下假设：

> {h.statement}

采用{d.intervention_type}干预，重复 {d.n_repeats} 次实验。
贝叶斯因子分析表明 BF = {a.bayes_factor:.4f}（{bf_interpretation}），
观测到干预效应大小为 {a.effect_size:.4f}。
**结论：假设被{decision_cn}**，置信度 {a.confidence:.2%}。

## 1. 引言

本研究由 ZeroDataModel 自主科学发现引擎生成。
假设来源策略：**{h.strategy}**。

- 初始置信度：{h.confidence:.2%}
- 可测试性评分：{h.testability:.2%}
- 零假设：{h.null_statement}

## 2. 方法

### 2.1 实验设计

- 干预类型：`{d.intervention_type}`
- 干预变量：{', '.join(d.intervention_vars) if d.intervention_vars else '无'}
- 观测变量：{', '.join(d.observation_vars) if d.observation_vars else '无'}
- 干预参数：{d.params}
- 重复次数：{d.n_repeats}
- 预期信息增益（EIG）：{d.expected_info_gain:.4f}

### 2.2 动作序列

{self._render_action_sequence(d.action_sequence)}

### 2.3 分析方法

采用贝叶斯因子分析：
- BF = P(D|H1) / P(D|H0)
- 接受阈值：BF > 10.0
- 拒绝阈值：BF < 0.1

## 3. 结果

- **贝叶斯因子**: {a.bayes_factor:.4f}
- **效应大小**: {a.effect_size:.4f}
- **样本数**: {a.n_samples}
- **置信度**: {a.confidence:.2%}
- **决策**: {decision_cn}

### 自然语言结论

{a.conclusion}

## 4. 讨论

本发现基于自主科学发现引擎的闭环流程（假设→实验→分析→发表）。
假设采用 **{h.strategy}** 策略生成。

{self._render_discussion(h, a)}

## 5. 知识图谱更新

{self._render_kg_delta(a.kg_delta)}

## 6. 未来工作

- 在更多场景下验证该发现的可重复性
- 探索 {', '.join(h.observation_vars) if h.observation_vars else '相关变量'} 的深层机制
- 与已有知识库中相关定律对比

---

*本论文由 ZeroDataModel 自主科学发现引擎自动生成，未经人类审阅。*
"""

    def _render_action_sequence(self, actions: list[dict]) -> str:
        if not actions:
            return "（无动作序列）"
        lines = []
        for a in actions:
            step = a.get("step", "?")
            action = a.get("action", "?")
            desc = a.get("description", "")
            params = a.get("params", {})
            line = f"{step}. **{action}**"
            if desc:
                line += f" — {desc}"
            if params:
                line += f"  \n   参数: `{params}`"
            lines.append(line)
        return "\n".join(lines)

    def _render_discussion(self, h: Hypothesis, a: AnalysisResult) -> str:
        if a.decision == "accept":
            return f"假设被接受，表明「{h.statement}」在当前实验条件下成立。这扩展了模型对{', '.join(h.intervention_vars) if h.intervention_vars else '相关变量'}的理解。"
        if a.decision == "reject":
            return f"假设被拒绝，表明「{h.statement}」在当前实验条件下不成立。模型应更新对{', '.join(h.intervention_vars) if h.intervention_vars else '相关变量'}的认知。"
        return "实验结果不确定，需要更多数据或更精细的实验设计来验证该假设。"

    def _render_kg_delta(self, kg_delta: dict) -> str:
        new_nodes = kg_delta.get("new_nodes", [])
        new_edges = kg_delta.get("new_edges", [])
        if not new_nodes and not new_edges:
            return "无知识图谱更新。"
        lines = []
        if new_nodes:
            lines.append(f"**新增节点**: {', '.join(map(str, new_nodes))}")
        if new_edges:
            lines.append("**新增边**:")
            for e in new_edges:
                lines.append(
                    f"  - {e.get('source')} → {e.get('target')} "
                    f"(weight={e.get('weight', 0)}, type={e.get('type', 'causal')})"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def papers(self) -> list[Paper]:
        return list(self._papers)

    @property
    def paper_count(self) -> int:
        return len(self._papers)

    def get_paper(self, paper_id: str) -> Paper | None:
        for p in self._papers:
            if p.paper_id == paper_id:
                return p
        return None

    def snapshot(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "paper_count": len(self._papers),
            "recent_titles": [p.title[:50] for p in self._papers[-5:]],
            "recent_decisions": [p.decision for p in self._papers[-5:]],
        }
