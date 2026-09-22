"""Discovery — 自主科学发现子包。

包含文献挖掘、假设生成、实验设计、结果分析、论文撰写、同行评审和发现
闭环控制器。
"""

from .hypothesis_generator import Hypothesis, HypothesisGenerator
from .experiment_designer import (
    ExperimentDesign,
    ExperimentDesigner,
    SandboxInterface,
)
from .result_analyzer import AnalysisResult, ResultAnalyzer
from .paper_writer import Paper, PaperWriter
from .literature_miner import (
    LiteratureMiner,
    LiteratureGraph,
    Paper as LiteraturePaper,
)
from .peer_reviewer import PeerReviewer, ReviewReport
from .science_loop import ScienceLoop, DiscoveryCycle

__all__ = [
    # 假设生成
    "Hypothesis",
    "HypothesisGenerator",
    # 实验设计
    "ExperimentDesign",
    "ExperimentDesigner",
    "SandboxInterface",
    # 结果分析
    "AnalysisResult",
    "ResultAnalyzer",
    # 论文撰写
    "Paper",
    "PaperWriter",
    # 文献挖掘
    "LiteratureMiner",
    "LiteratureGraph",
    "LiteraturePaper",
    # 同行评审
    "PeerReviewer",
    "ReviewReport",
    # 科学发现闭环
    "ScienceLoop",
    "DiscoveryCycle",
]
