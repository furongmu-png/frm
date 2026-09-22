"""认知技能单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from skills.base import SkillContext
from skills.cognition.theory_of_mind import TheoryOfMindModule, AgentModel
from skills.cognition.narrative import NarrativeModule
from skills.cognition.affect import AffectModule, AFFECT_CATEGORIES
from skills.cognition.analogy import AnalogyEngine, ConceptGraph


# ================================================================== #
# 5. TheoryOfMindModule
# ================================================================== #
class TestAgentModel:
    def test_observe_updates_belief(self):
        model = AgentModel(agent_id=1, belief_dim=16)
        obs = np.ones(16) * 0.5
        model.observe(obs, None)
        assert np.allclose(model.estimated_belief, np.ones(16) * 0.05, atol=0.01)

    def test_predict_next_action(self):
        model = AgentModel(agent_id=1, belief_dim=8)
        obs = np.ones(8) * 0.5
        action = np.ones(8) * 0.3
        model.observe(obs, action)
        pred = model.predict_next_action()
        assert pred.shape == (8,)
        assert np.isfinite(pred).all()

    def test_goal_inference(self):
        model = AgentModel(agent_id=1, belief_dim=8)
        # 重复同一动作（零趋势） → stationary
        for _ in range(5):
            model.observe(np.ones(8) * 0.5, np.ones(8) * 0.3)
        assert model.infer_goal() == "stationary"

    def test_goal_inference_exploring(self):
        model = AgentModel(agent_id=1, belief_dim=8)
        # 高方差动作 → exploring
        rng = np.random.default_rng(7)
        for i in range(5):
            model.observe(np.ones(8) * 0.5, rng.standard_normal(8) * 0.5)
        assert model.infer_goal() == "exploring"

    def test_prediction_error_decreases(self):
        """重复观测后预测误差应下降。"""
        model = AgentModel(agent_id=1, belief_dim=8)
        obs = np.ones(8) * 0.5
        action = np.ones(8) * 0.3
        model.observe(obs, action)
        err1 = model._last_prediction_error
        model.observe(obs, action)
        err2 = model._last_prediction_error
        # 误差应有所下降（模型在学习）
        assert err2 <= err1


class TestTheoryOfMindModule:
    def test_single_agent_mode(self):
        """无多智能体世界时，应模拟虚拟其他智能体。"""
        tom = TheoryOfMindModule(belief_dim=16)
        ctx = SkillContext(belief=np.zeros(64))
        result = tom.safe_process(ctx)
        assert result.error is None
        assert result.data["n_modeled_agents"] >= 1

    def test_observing_multiple_steps(self):
        tom = TheoryOfMindModule(belief_dim=16)
        for i in range(5):
            ctx = SkillContext(belief=np.random.default_rng(i).standard_normal(64))
            tom.safe_process(ctx)
        assert tom._agent_models  # 有模型
        result = tom.safe_process(SkillContext(belief=np.zeros(64)))
        assert "predictions" in result.data


# ================================================================== #
# 6. NarrativeModule
# ================================================================== #
class TestNarrativeModule:
    def test_ingest_and_parse(self):
        narr = NarrativeModule()
        text = "Alice walked to the forest. She saw a rabbit. But the rabbit ran away. Alice was sad."
        narr.ingest_text(text)
        structure = narr.parse_structure(text)
        assert "characters" in structure
        assert "conflicts" in structure
        assert structure["n_sentences"] > 0

    def test_continue_story(self):
        narr = NarrativeModule()
        text = "The hero entered the dark cave. Suddenly a dragon appeared. The hero fought bravely."
        narr.ingest_text(text)
        continuation = narr.continue_story(seed="The hero", max_tokens=20)
        assert isinstance(continuation, str)

    def test_coherence_evaluation(self):
        narr = NarrativeModule()
        for text in ["The sun rose.", "The sun set.", "The moon rose."]:
            narr.ingest_text(text)
        coherence = narr.evaluate_coherence()
        assert 0.0 < coherence <= 1.0

    def test_process_via_context(self):
        narr = NarrativeModule()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation="Bob went home. He was happy. But then it rained.",
        )
        result = narr.safe_process(ctx)
        assert result.error is None
        assert "structure" in result.data
        assert "continuation" in result.data
        assert "coherence" in result.data


# ================================================================== #
# 7. AffectModule
# ================================================================== #
class TestAffectModule:
    def test_positive_sentiment(self):
        aff = AffectModule()
        category, probs = aff.classify_sentiment("This is great and wonderful!")
        assert category == "positive"
        assert probs[0] > probs[1]

    def test_negative_sentiment(self):
        aff = AffectModule()
        category, probs = aff.classify_sentiment("This is terrible and awful.")
        assert category == "negative"
        assert probs[1] > probs[0]

    def test_neutral_sentiment(self):
        aff = AffectModule()
        category, probs = aff.classify_sentiment("The table is there.")
        assert category == "neutral"

    def test_humor_detection(self):
        aff = AffectModule()
        humor = aff.detect_humor("That was so funny lol haha!")
        assert humor > 0.0

    def test_no_humor(self):
        aff = AffectModule()
        humor = aff.detect_humor("The weather is cloudy today.")
        assert humor == 0.0

    def test_affect_vector_shape(self):
        aff = AffectModule(latent_dim=8)
        _, probs = aff.classify_sentiment("happy good great")
        vec = aff.generate_affect_vector(probs, 0.5)
        assert vec.shape == (8,)

    def test_preference_prior_sign(self):
        """积极情感应产生负偏好（降低自由能）。"""
        aff = AffectModule(latent_dim=8)
        _, probs = aff.classify_sentiment("happy wonderful great")
        prior = aff.compute_preference_prior(probs)
        assert prior < 0  # 积极趋近

    def test_process_via_context(self):
        aff = AffectModule()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation="This is amazing and wonderful!",
        )
        result = aff.safe_process(ctx)
        assert result.error is None
        assert result.data["sentiment"] == "positive"
        assert "humor" in result.data
        assert "affect_vector" in result.data


# ================================================================== #
# 8. AnalogyEngine
# ================================================================== #
class TestConceptGraph:
    def test_add_and_query(self):
        g = ConceptGraph()
        g.add("sun", "illuminates", "earth")
        g.add("lamp", "illuminates", "room")
        matches = g.find_structural_matches(
            "sun", [("illuminates", "earth")]
        )
        assert any(m[0] == "lamp" for m in matches)


class TestAnalogyEngine:
    def test_find_analogy(self):
        engine = AnalogyEngine()
        result = engine.find_analogy("logic", "structures", "argument")
        assert result is not None
        assert "sentence" in result
        assert "analog" in result
        assert result["similarity"] > 0

    def test_analogy_sentence_format(self):
        engine = AnalogyEngine()
        result = engine.find_analogy("memory", "stores", "knowledge")
        assert result is not None
        assert "之于" in result["sentence"]
        assert "就像" in result["sentence"]

    def test_add_custom_concept(self):
        engine = AnalogyEngine()
        engine.add_concept("river", "flows", "valley")
        engine.add_concept("road", "flows", "city")
        result = engine.find_analogy("river", "flows", "valley")
        assert result is not None

    def test_process_via_context(self):
        engine = AnalogyEngine()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation={"concept": "logic", "relation": "structures", "object": "argument"},
        )
        result = engine.safe_process(ctx)
        assert result.error is None
        assert result.data["found"] is True
        assert "sentence" in result.data

    def test_no_match_returns_none(self):
        engine = AnalogyEngine()
        result = engine.find_analogy("nonexistent_concept", "relates", "nothing")
        assert result is None
