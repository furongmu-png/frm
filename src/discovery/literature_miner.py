"""LiteratureMiner — mines scientific literature and builds a knowledge graph.

理论基础
========
科学发现的第一步是文献调研。本模块连接公开论文 API（arXiv、Semantic Scholar）
或从本地文献库读取，对每篇论文提取标题/摘要/方法/结论，用 JEPA 目标编码器
得到嵌入，构建科学知识图谱（节点=论文/概念/方法，边=引用/相似度/矛盾）。

该图谱与模型已有知识图谱融合，成为科学发现的先验。

设计原则：
  - 无网络时降级为本地文献库（默认空）
  - 嵌入用确定性哈希投影（无外部 NLP 依赖，与 ImagePreprocessor 一致风格）
  - 知识图谱结构与现有 knowledge_graph 兼容（nodes + edges）
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class Paper:
    """一篇文献。"""
    paper_id: str
    title: str
    abstract: str
    methods: str = ""
    conclusion: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    url: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(64))
    source: str = "unknown"          # "arxiv" | "semantic_scholar" | "local"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiteratureGraph:
    """科学知识图谱。"""
    papers: list[Paper] = field(default_factory=list)
    # 图结构（与现有 knowledge_graph 兼容）
    nodes: list[dict] = field(default_factory=list)   # {id, type, label, ...}
    edges: list[dict] = field(default_factory=list)  # {source, target, type, weight}
    # 概念索引（概念 → 论文 ID 列表）
    concept_index: dict[str, list[str]] = field(default_factory=dict)
    # 矛盾对（相互矛盾的论文对）
    contradictions: list[tuple[str, str, float]] = field(default_factory=list)


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #
class LiteratureMiner:
    """文献挖掘与知识图谱构建器。

    Parameters
    ----------
    embedding_dim : int
        嵌入维度（与模型隐空间一致）
    cache_dir : str
        本地文献缓存目录
    seed : int | None
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        cache_dir: str = "literature_cache",
        seed: Optional[int] = 42,
    ):
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")

        self.embedding_dim = int(embedding_dim)
        self.cache_dir = cache_dir
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

        # 确定性嵌入投影矩阵（Johnson-Lindenstrauss 风格）
        # 用文本哈希 → 确定性索引 → 投影
        self._proj_matrix = self._rng.standard_normal(
            (embedding_dim, 256)
        ) * np.sqrt(1.0 / 256)

        self.graph = LiteratureGraph()
        self._paper_index: dict[str, Paper] = {}

        # 创建缓存目录
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            pass

    # ---------------------------------------------------------------- #
    # 文本嵌入（确定性，无外部依赖）
    # ---------------------------------------------------------------- #
    def embed_text(self, text: str) -> np.ndarray:
        """将文本编码为嵌入向量。

        用确定性哈希将文本映射到 256 维稀疏向量，再投影到 embedding_dim。
        """
        if not text:
            return np.zeros(self.embedding_dim)
        # 哈希分词（简单：按字符 n-gram）
        tokens = self._tokenize(text)
        # 构建稀疏向量
        sparse = np.zeros(256)
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % 256
            sparse[h] += 1.0
        # 归一化
        norm = np.linalg.norm(sparse)
        if norm > 0:
            sparse = sparse / norm
        # 投影
        emb = self._proj_matrix @ sparse
        # tanh 压缩
        emb = np.tanh(emb)
        if not np.all(np.isfinite(emb)):
            emb = np.zeros(self.embedding_dim)
        return emb

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：英文按空格+标点，中文按字。"""
        tokens: list[str] = []
        # 英文词
        current = ""
        for ch in text.lower():
            if ch.isalnum():
                current += ch
            else:
                if current:
                    tokens.append(current)
                    current = ""
                # 中文字符作为单 token
                if "\u4e00" <= ch <= "\u9fff":
                    tokens.append(ch)
        if current:
            tokens.append(current)
        return tokens

    # ---------------------------------------------------------------- #
    # 从 arXiv API 检索（无网络时降级）
    # ---------------------------------------------------------------- #
    def search_arxiv(
        self,
        query: str,
        max_results: int = 5,
        timeout: float = 5.0,
    ) -> list[Paper]:
        """从 arXiv API 检索论文（简化，无 XML 解析依赖）。

        arXiv API: http://export.arxiv.org/api/query?search_query=...
        无网络时返回空列表。
        """
        try:
            url = (
                "http://export.arxiv.org/api/query?search_query="
                + urllib.parse.quote(query)
                + f"&max_results={max_results}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "ZeroDataModel/0.2"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
            return self._parse_arxiv_response(content)
        except Exception:
            return []

    def _parse_arxiv_response(self, content: str) -> list[Paper]:
        """简单解析 arXiv Atom XML 响应（无外部 XML 库）。"""
        papers: list[Paper] = []
        # 按 <entry> 分割
        entries = content.split("<entry>")[1:]
        for entry in entries:
            entry = entry.split("</entry>")[0]
            try:
                title = self._extract_xml(entry, "title")
                summary = self._extract_xml(entry, "summary")
                paper_id = self._extract_xml(entry, "id")
                published = self._extract_xml(entry, "published")
                year = 0
                if published and len(published) >= 4:
                    year = int(published[:4])
                if title and summary:
                    p = Paper(
                        paper_id=hashlib.md5(paper_id.encode()).hexdigest()[:12],
                        title=title.strip(),
                        abstract=summary.strip(),
                        year=year,
                        url=paper_id.strip(),
                        source="arxiv",
                    )
                    p.embedding = self.embed_text(title + " " + summary)
                    papers.append(p)
            except Exception:
                continue
        return papers

    @staticmethod
    def _extract_xml(text: str, tag: str) -> str:
        """简单提取 XML 标签内容。"""
        start = text.find(f"<{tag}")
        if start < 0:
            return ""
        # 跳过属性
        gt = text.find(">", start)
        if gt < 0:
            return ""
        end = text.find(f"</{tag}>", gt)
        if end < 0:
            return ""
        return text[gt + 1 : end]

    # ---------------------------------------------------------------- #
    # 从本地文献库加载
    # ---------------------------------------------------------------- #
    def load_local_papers(self, papers: list[dict[str, Any]]) -> list[Paper]:
        """从本地文献列表加载。

        Parameters
        ----------
        papers : list of dict
            每个字典含 title, abstract, methods, conclusion, authors, year
        """
        result: list[Paper] = []
        with self._lock:
            for p in papers:
                if not isinstance(p, dict):
                    continue
                title = str(p.get("title", "")).strip()
                abstract = str(p.get("abstract", "")).strip()
                if not title and not abstract:
                    continue
                paper_id = hashlib.md5(
                    (title + abstract).encode("utf-8")
                ).hexdigest()[:12]
                paper = Paper(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    methods=str(p.get("methods", "")),
                    conclusion=str(p.get("conclusion", "")),
                    authors=list(p.get("authors", [])),
                    year=int(p.get("year", 0)),
                    url=str(p.get("url", "")),
                    source="local",
                )
                paper.embedding = self.embed_text(title + " " + abstract)
                self._paper_index[paper_id] = paper
                self.graph.papers.append(paper)
                result.append(paper)
        return result

    # ---------------------------------------------------------------- #
    # 构建知识图谱
    # ---------------------------------------------------------------- #
    def build_graph(self, similarity_threshold: float = 0.3) -> LiteratureGraph:
        """构建科学知识图谱。

        - 节点：每篇论文 + 提取的概念
        - 边：相似度（余弦）、引用、矛盾
        """
        with self._lock:
            self.graph.nodes.clear()
            self.graph.edges.clear()
            self.graph.concept_index.clear()
            self.graph.contradictions.clear()

            # 论文节点
            for paper in self.graph.papers:
                self.graph.nodes.append({
                    "id": paper.paper_id,
                    "type": "paper",
                    "label": paper.title[:60],
                    "year": paper.year,
                    "source": paper.source,
                })
                # 概念提取（简单：标题中的关键词）
                concepts = self._extract_concepts(paper.title + " " + paper.abstract)
                for concept in concepts:
                    self.graph.concept_index.setdefault(concept, []).append(
                        paper.paper_id
                    )
                    self.graph.nodes.append({
                        "id": f"concept_{concept}",
                        "type": "concept",
                        "label": concept,
                    })
                    # 论文 → 概念 边
                    self.graph.edges.append({
                        "source": paper.paper_id,
                        "target": f"concept_{concept}",
                        "type": "mentions",
                        "weight": 1.0,
                    })

            # 论文间相似度边
            n = len(self.graph.papers)
            for i in range(n):
                for j in range(i + 1, n):
                    pi = self.graph.papers[i]
                    pj = self.graph.papers[j]
                    sim = float(self._cosine_similarity(
                        pi.embedding, pj.embedding
                    ))
                    if sim > similarity_threshold:
                        self.graph.edges.append({
                            "source": pi.paper_id,
                            "target": pj.paper_id,
                            "type": "similar",
                            "weight": sim,
                        })
                    # 检测矛盾（高相似但结论相反——简化：用随机扰动）
                    elif sim > 0.2 and self._detect_contradiction(pi, pj):
                        self.graph.edges.append({
                            "source": pi.paper_id,
                            "target": pj.paper_id,
                            "type": "contradiction",
                            "weight": 1.0 - sim,
                        })
                        self.graph.contradictions.append(
                            (pi.paper_id, pj.paper_id, 1.0 - sim)
                        )

            return self.graph

    def _extract_concepts(self, text: str) -> list[str]:
        """从文本提取概念（简化：高频词）。"""
        tokens = self._tokenize(text)
        # 过滤停用词
        stopwords = {"the", "a", "an", "of", "in", "on", "and", "or", "to", "is", "are", "for", "with", "by"}
        freq: dict[str, int] = {}
        for t in tokens:
            if len(t) > 2 and t not in stopwords:
                freq[t] = freq.get(t, 0) + 1
        # 取前 5 个高频词
        sorted_concepts = sorted(freq.items(), key=lambda x: -x[1])[:5]
        return [c for c, _ in sorted_concepts]

    def _detect_contradiction(self, p1: Paper, p2: Paper) -> bool:
        """简化矛盾检测：标题/结论中包含对立词。"""
        opposite_pairs = [
            ("increase", "decrease"), ("positive", "negative"),
            ("yes", "no"), ("true", "false"), ("high", "low"),
        ]
        t1 = (p1.title + " " + p1.conclusion).lower()
        t2 = (p2.title + " " + p2.conclusion).lower()
        for a, b in opposite_pairs:
            if (a in t1 and b in t2) or (b in t1 and a in t2):
                return True
        return False

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ---------------------------------------------------------------- #
    # 融合到现有知识图谱
    # ---------------------------------------------------------------- #
    def merge_into(
        self, existing_kg: dict[str, Any]
    ) -> dict[str, Any]:
        """将科学知识图谱融合到现有模型知识图谱。

        Returns
        -------
        dict
            融合后的知识图谱
        """
        with self._lock:
            if not isinstance(existing_kg, dict):
                existing_kg = {"nodes": [], "edges": []}
            existing_nodes = list(existing_kg.get("nodes", []))
            existing_edges = list(existing_kg.get("edges", []))

            # 追加新节点（去重）
            existing_ids = {n.get("id") for n in existing_nodes if isinstance(n, dict)}
            for node in self.graph.nodes:
                if node.get("id") not in existing_ids:
                    existing_nodes.append(node)
                    existing_ids.add(node.get("id"))

            # 追加新边
            existing_edge_keys = {
                (e.get("source"), e.get("target"))
                for e in existing_edges if isinstance(e, dict)
            }
            for edge in self.graph.edges:
                key = (edge.get("source"), edge.get("target"))
                if key not in existing_edge_keys:
                    existing_edges.append(edge)
                    existing_edge_keys.add(key)

            return {
                "nodes": existing_nodes,
                "edges": existing_edges,
                "n_papers": len(self.graph.papers),
                "n_contradictions": len(self.graph.contradictions),
            }

    # ---------------------------------------------------------------- #
    # 查询
    # ---------------------------------------------------------------- #
    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "n_papers": len(self.graph.papers),
                "n_nodes": len(self.graph.nodes),
                "n_edges": len(self.graph.edges),
                "n_concepts": len(self.graph.concept_index),
                "n_contradictions": len(self.graph.contradictions),
                "sources": list({p.source for p in self.graph.papers}),
            }

    def find_structural_holes(self) -> list[tuple[str, str, float]]:
        """发现知识图谱的结构空洞（稀疏连接区域）——跨学科链接机会。

        Returns
        -------
        list of (concept_a, concept_b, opportunity_score)
        """
        with self._lock:
            concepts = list(self.graph.concept_index.keys())
            if len(concepts) < 2:
                return []
            # 计算概念间的共现频率
            holes: list[tuple[str, str, float]] = []
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    ca, cb = concepts[i], concepts[j]
                    papers_a = set(self.graph.concept_index[ca])
                    papers_b = set(self.graph.concept_index[cb])
                    cooccur = len(papers_a & papers_b)
                    # 共现少但各自都有论文 → 结构空洞
                    if cooccur == 0 and len(papers_a) > 0 and len(papers_b) > 0:
                        opportunity = 1.0 / (1.0 + abs(len(papers_a) - len(papers_b)))
                        holes.append((ca, cb, opportunity))
            holes.sort(key=lambda x: -x[2])
            return holes[:10]

    def reset(self) -> None:
        with self._lock:
            self.graph = LiteratureGraph()
            self._paper_index.clear()
