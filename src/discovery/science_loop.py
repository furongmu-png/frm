"""Discovery Science Loop — 自主科学发现闭环控制器。

每 N 步自动触发发现循环：
  1. 文献扫描（升级 3.1）
  2. 知识图谱更新
  3. 假设生成（含文献驱动，升级 3.2）
  4. 实验设计
  5. 执行实验（在沙盒中）
  6. 分析结果
  7. 撰写论文
  8. 内部同行评审（新增 3.5）
  9. 更新知识库

人类可通过前端审批实验（可选），或完全自动运行。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .hypothesis_generator import HypothesisGenerator, Hypothesis
from .experiment_designer import ExperimentDesigner, ExperimentDesign, SandboxInterface
from .result_analyzer import ResultAnalyzer, AnalysisResult
from .paper_writer import PaperWriter, Paper
from .literature_miner import LiteratureMiner, LiteratureGraph
from .peer_reviewer import PeerReviewer, ReviewReport


@dataclass
class DiscoveryCycle:
    """一次完整的发现循环记录。"""

    cycle_id: int
    timestamp: float
    n_hypotheses: int
    n_experiments: int
    n_accepted: int
    n_rejected: int
    papers_written: int
    #: 关键发现摘要
    findings: list[str] = field(default_factory=list)
    #: 错误信息（若有）
    error: str | None = None
    #: 文献扫描统计（升级 3.1）
    n_papers_scanned: int = 0
    n_contradictions_found: int = 0
    n_structural_holes: int = 0
    #: 同行评审统计（升级 3.5）
    n_reviews: int = 0
    n_review_accept: int = 0
    n_review_revise: int = 0
    n_review_reject: int = 0


class ScienceLoop:
    """科学发现闭环控制器。

    Parameters
    ----------
    trigger_interval : int, default 5000
        每多少步触发一次发现循环。
    output_dir : str, default "discoveries"
        论文存档目录。
    require_approval : bool, default False
        True: 实验需人类审批；False: 完全自动。
    enable_literature_mining : bool, default True
        是否启用文献挖掘步骤（升级 3.1）。
    enable_peer_review : bool, default True
        是否启用内部同行评审（升级 3.5）。
    seed : int, default 42
    """

    def __init__(
        self,
        trigger_interval: int = 5000,
        output_dir: str = "discoveries",
        require_approval: bool = False,
        enable_literature_mining: bool = True,
        enable_peer_review: bool = True,
        seed: int = 42,
    ) -> None:
        if trigger_interval <= 0:
            raise ValueError(f"trigger_interval must be positive, got {trigger_interval}")

        self.trigger_interval = int(trigger_interval)
        self.require_approval = bool(require_approval)
        self.enable_literature_mining = bool(enable_literature_mining)
        self.enable_peer_review = bool(enable_peer_review)

        self.hypothesis_generator = HypothesisGenerator(seed=seed)
        self.experiment_designer = ExperimentDesigner(seed=seed + 1)
        self.result_analyzer = ResultAnalyzer(seed=seed + 2)
        self.paper_writer = PaperWriter(output_dir=output_dir)
        # 升级 3.1：文献挖掘器
        self.literature_miner = LiteratureMiner(
            embedding_dim=64,
            cache_dir=f"{output_dir}/literature_cache",
            seed=seed + 3,
        )
        # 升级 3.5：内部同行评审器
        self.peer_reviewer = PeerReviewer(
            accept_threshold=0.7,
            reject_threshold=0.3,
            seed=seed + 4,
        )

        self._enabled = True
        self._step = 0
        self._cycle_count = 0
        self._cycles: list[DiscoveryCycle] = []
        self._max_cycles = 50
        #: 待审批的实验队列
        self._pending_approvals: list[ExperimentDesign] = []
        #: 上次循环的快照缓存（供前端轮询）
        self._last_snapshot: dict[str, Any] = {}
        #: 累积文献图谱快照（升级 3.1，跨循环累积）
        self._accumulated_lit_graph: dict[str, Any] = {
            "papers": [],
            "nodes": [],
            "edges": [],
            "concept_index": {},
            "contradictions": [],
        }

    # ------------------------------------------------------------------ #
    # 沙盒附加
    # ------------------------------------------------------------------ #
    def attach_sandbox(self, sandbox: SandboxInterface) -> None:
        """附加物理沙盒以执行真实实验。"""
        self.experiment_designer.attach_sandbox(sandbox)

    def detach_sandbox(self) -> None:
        self.experiment_designer.detach_sandbox()

    # ------------------------------------------------------------------ #
    # 文献挖掘接口（升级 3.1）
    # ------------------------------------------------------------------ #
    def load_local_papers(self, papers: list[dict[str, Any]]) -> int:
        """加载本地文献库。

        Returns
        -------
        int
            实际加载的论文数量。
        """
        if not self.enable_literature_mining:
            return 0
        loaded = self.literature_miner.load_local_papers(papers)
        return len(loaded)

    def search_literature(self, query: str, max_results: int = 5) -> int:
        """在线检索文献（arXiv API）。

        Returns
        -------
        int
            实际检索到的论文数量。
        """
        if not self.enable_literature_mining:
            return 0
        papers = self.literature_miner.search_arxiv(query, max_results=max_results)
        # 加入图谱
        for p in papers:
            self.literature_miner.graph.papers.append(p)
            self.literature_miner._paper_index[p.paper_id] = p
        return len(papers)

    def register_known_finding(self, finding: str) -> None:
        """注册已知文献结论，用于评审时一致性检查。"""
        self.peer_reviewer.register_known_finding(finding)

    # ------------------------------------------------------------------ #
    # 主循环：每步调用
    # ------------------------------------------------------------------ #
    def step(
        self,
        causal_graph: dict | None = None,
        knowledge_graph: dict | None = None,
        prediction_errors: dict[str, float] | None = None,
        logic_contradictions: list[dict] | None = None,
    ) -> DiscoveryCycle | None:
        """推进一步，按触发间隔执行发现循环。

        Returns
        -------
        DiscoveryCycle | None
            若本步触发了循环，返回循环记录；否则 None。
        """
        if not self._enabled:
            return None

        self._step += 1
        if self._step % self.trigger_interval != 0:
            return None

        return self.run_cycle(
            causal_graph=causal_graph,
            knowledge_graph=knowledge_graph,
            prediction_errors=prediction_errors,
            logic_contradictions=logic_contradictions,
        )

    def run_cycle(
        self,
        causal_graph: dict | None = None,
        knowledge_graph: dict | None = None,
        prediction_errors: dict[str, float] | None = None,
        logic_contradictions: list[dict] | None = None,
    ) -> DiscoveryCycle:
        """立即执行一次完整的发现循环。"""
        cycle_id = self._cycle_count
        self._cycle_count += 1
        timestamp = time.time()
        findings: list[str] = []
        error: str | None = None

        # 升级统计字段
        n_papers_scanned = 0
        n_contradictions_found = 0
        n_structural_holes = 0
        n_reviews = 0
        n_review_accept = 0
        n_review_revise = 0
        n_review_reject = 0

        try:
            # 0. 文献扫描与知识图谱更新（升级 3.1）
            lit_contradictions: list[tuple] = []
            structural_holes: list[tuple] = []
            lit_graph_dict: dict[str, Any] = {}

            if self.enable_literature_mining and self.literature_miner.graph.papers:
                lit_graph = self.literature_miner.build_graph()
                n_papers_scanned = len(lit_graph.papers)
                n_contradictions_found = len(lit_graph.contradictions)
                lit_contradictions = list(lit_graph.contradictions)
                structural_holes = self.literature_miner.find_structural_holes()
                n_structural_holes = len(structural_holes)

                # 累积到全局知识图谱
                if knowledge_graph is None:
                    knowledge_graph = {"nodes": [], "edges": []}
                knowledge_graph = self.literature_miner.merge_into(knowledge_graph)

                # 文献图谱快照（用于假设生成）
                lit_graph_dict = {
                    "papers": [
                        {
                            "paper_id": p.paper_id,
                            "title": p.title,
                            "abstract": p.abstract,
                            "year": p.year,
                            "source": p.source,
                        }
                        for p in lit_graph.papers
                    ],
                    "nodes": lit_graph.nodes,
                    "edges": lit_graph.edges,
                    "concept_index": lit_graph.concept_index,
                    "contradictions": lit_contradictions,
                }
                self._accumulated_lit_graph = lit_graph_dict

                if n_papers_scanned > 0:
                    findings.append(
                        f"扫描 {n_papers_scanned} 篇文献，"
                        f"发现 {n_contradictions_found} 处矛盾，"
                        f"{n_structural_holes} 个结构空洞"
                    )

            # 1. 生成假设（传统策略）
            hypotheses = self.hypothesis_generator.generate(
                causal_graph=causal_graph,
                knowledge_graph=knowledge_graph,
                prediction_errors=prediction_errors,
                logic_contradictions=logic_contradictions,
            )

            # 1b. 文献驱动假设（升级 3.2）
            if self.enable_literature_mining and lit_graph_dict:
                lit_hypotheses = self.hypothesis_generator.generate_from_literature(
                    literature_graph=lit_graph_dict,
                    contradictions=lit_contradictions,
                    structural_holes=structural_holes,
                )
                # 合并并去重（按 statement）
                existing_stmts = {h.statement for h in hypotheses}
                for h in lit_hypotheses:
                    if h.statement not in existing_stmts:
                        hypotheses.append(h)
                        existing_stmts.add(h.statement)

            # 2. 设计实验
            designs = self.experiment_designer.design_batch(hypotheses)

            # 3. 执行实验（若需审批则入队）
            n_accepted = 0
            n_rejected = 0
            papers_written = 0
            for design in designs:
                if self.require_approval:
                    self._pending_approvals.append(design)
                    continue

                # 执行
                exp_data = self.experiment_designer.execute(design)

                # 找到对应假设
                hyp = next(
                    (h for h in hypotheses if h.hypothesis_id == design.hypothesis_id),
                    None,
                )
                if hyp is None:
                    continue

                # 4. 分析结果
                analysis = self.result_analyzer.analyze(hyp, exp_data)

                # 5. 撰写论文
                if analysis.decision in ("accept", "reject"):
                    paper = self.paper_writer.write(hyp, design, analysis)
                    papers_written += 1
                    findings.append(
                        f"{analysis.decision.upper()}: {hyp.statement[:80]}"
                    )

                    # 6. 内部同行评审（升级 3.5）
                    if self.enable_peer_review:
                        review = self.peer_reviewer.review(
                            paper=paper,
                            hypothesis=hyp,
                            design=design,
                            analysis=analysis,
                        )
                        n_reviews += 1
                        if review.decision == "accept":
                            n_review_accept += 1
                            findings.append(
                                f"评审通过：{hyp.statement[:60]}"
                            )
                        elif review.decision == "revise":
                            n_review_revise += 1
                            findings.append(
                                f"评审需修订：{review.required_revisions[:1]}"
                            )
                        else:
                            n_review_reject += 1

                if analysis.decision == "accept":
                    n_accepted += 1
                elif analysis.decision == "reject":
                    n_rejected += 1

            cycle = DiscoveryCycle(
                cycle_id=cycle_id,
                timestamp=timestamp,
                n_hypotheses=len(hypotheses),
                n_experiments=len(designs),
                n_accepted=n_accepted,
                n_rejected=n_rejected,
                papers_written=papers_written,
                findings=findings,
                n_papers_scanned=n_papers_scanned,
                n_contradictions_found=n_contradictions_found,
                n_structural_holes=n_structural_holes,
                n_reviews=n_reviews,
                n_review_accept=n_review_accept,
                n_review_revise=n_review_revise,
                n_review_reject=n_review_reject,
            )
        except Exception as exc:
            # H5 修复：从 (ValueError, KeyError, RuntimeError, OSError)
            # 拓宽到 Exception。原代码会漏掉 TypeError/AttributeError/
            # IndexError/ZeroDivisionError 等，导致 cycle 变量未绑定
            # （UnboundLocalError）并向上传播，中断 think()。
            # 军事级健壮性要求：任何异常都应记录为 cycle.error，
            # 而非让整个发现循环崩溃。
            error = f"{type(exc).__name__}: {exc}"
            cycle = DiscoveryCycle(
                cycle_id=cycle_id,
                timestamp=timestamp,
                n_hypotheses=0,
                n_experiments=0,
                n_accepted=0,
                n_rejected=0,
                papers_written=0,
                error=error,
            )

        self._cycles.append(cycle)
        if len(self._cycles) > self._max_cycles:
            self._cycles = self._cycles[-self._max_cycles :]

        # 更新快照缓存
        self._last_snapshot = self.snapshot()

        return cycle

    # ------------------------------------------------------------------ #
    # 审批接口
    # ------------------------------------------------------------------ #
    def approve_pending(self, design_id: str) -> dict[str, Any] | None:
        """审批待执行的实验。

        F4 修复：原代码 execute() 无异常保护，沙盒执行抛异常时
        直接传播给调用方（Web 审批处理器），且 _pending_approvals
        未 pop 导致设计卡在队列中。现在包裹 try-except，失败时
        返回 {"error": ...} 并清理队列。
        """
        for i, design in enumerate(self._pending_approvals):
            if design.experiment_id == design_id:
                self._pending_approvals.pop(i)
                try:
                    return self.experiment_designer.execute(design)
                except Exception as exc:
                    return {"error": f"{type(exc).__name__}: {exc}"}
        return None

    def reject_pending(self, design_id: str) -> bool:
        """拒绝待执行的实验。"""
        for i, design in enumerate(self._pending_approvals):
            if design.experiment_id == design_id:
                self._pending_approvals.pop(i)
                return True
        return False

    @property
    def pending_approvals(self) -> list[ExperimentDesign]:
        return list(self._pending_approvals)

    # ------------------------------------------------------------------ #
    # 配置与快照
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def configure(self, **kwargs: Any) -> None:
        if "trigger_interval" in kwargs:
            v = int(kwargs["trigger_interval"])
            if v <= 0:
                raise ValueError("trigger_interval must be positive")
            self.trigger_interval = v
        if "require_approval" in kwargs:
            self.require_approval = bool(kwargs["require_approval"])
        if "enabled" in kwargs:
            self._enabled = bool(kwargs["enabled"])
        if "enable_literature_mining" in kwargs:
            self.enable_literature_mining = bool(kwargs["enable_literature_mining"])
        if "enable_peer_review" in kwargs:
            self.enable_peer_review = bool(kwargs["enable_peer_review"])

    @property
    def cycles(self) -> list[DiscoveryCycle]:
        return list(self._cycles)

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def total_papers(self) -> int:
        return self.paper_writer.paper_count

    def snapshot(self) -> dict[str, Any]:
        """返回前端可视化快照。"""
        recent_cycles = self._cycles[-5:] if self._cycles else []
        snap = {
            "enabled": self._enabled,
            "trigger_interval": self.trigger_interval,
            "step": self._step,
            "cycle_count": self._cycle_count,
            "total_papers": self.paper_writer.paper_count,
            "pending_approvals": len(self._pending_approvals),
            "require_approval": self.require_approval,
            "enable_literature_mining": self.enable_literature_mining,
            "enable_peer_review": self.enable_peer_review,
            "hypothesis_generator": self.hypothesis_generator.snapshot(),
            "experiment_designer": self.experiment_designer.snapshot(),
            "result_analyzer": self.result_analyzer.snapshot(),
            "paper_writer": self.paper_writer.snapshot(),
            "literature_miner": (
                self.literature_miner.get_snapshot()
                if self.enable_literature_mining else None
            ),
            "peer_reviewer": (
                self.peer_reviewer.get_snapshot()
                if self.enable_peer_review else None
            ),
            "recent_cycles": [
                {
                    "cycle_id": c.cycle_id,
                    "n_hypotheses": c.n_hypotheses,
                    "n_experiments": c.n_experiments,
                    "n_accepted": c.n_accepted,
                    "n_rejected": c.n_rejected,
                    "papers_written": c.papers_written,
                    "n_papers_scanned": c.n_papers_scanned,
                    "n_contradictions_found": c.n_contradictions_found,
                    "n_structural_holes": c.n_structural_holes,
                    "n_reviews": c.n_reviews,
                    "n_review_accept": c.n_review_accept,
                    "n_review_revise": c.n_review_revise,
                    "n_review_reject": c.n_review_reject,
                    "findings": c.findings[:3],
                    "error": c.error,
                }
                for c in recent_cycles
            ],
        }
        return snap

    @property
    def last_snapshot(self) -> dict[str, Any]:
        """返回最近一次循环的快照（无循环则返回空 dict）。"""
        return dict(self._last_snapshot) if self._last_snapshot else {}
