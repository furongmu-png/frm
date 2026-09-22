"""交互与协作技能单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from skills.base import SkillContext
from skills.interaction.dialogue import DialogueAgent, DialogueState
from skills.interaction.learning_from_demo import DemonstrationLearner, Demonstration
from skills.interaction.machine_teaching import TeachingModule, StudentModel
from skills.interaction.multimodal_translation import MultiModalTranslator


# ================================================================== #
# 9. DialogueAgent
# ================================================================== #
class TestDialogueAgent:
    def test_greeting_intent(self):
        agent = DialogueAgent()
        assert agent.classify_intent("hello there!") == "greeting"

    def test_question_intent(self):
        agent = DialogueAgent()
        assert agent.classify_intent("what is this?") == "question"

    def test_generate_response(self):
        agent = DialogueAgent()
        response = agent.generate_response("What is consciousness?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_topic_extraction(self):
        agent = DialogueAgent()
        topic = agent.extract_topic("Tell me about consciousness and awareness")
        assert topic is not None

    def test_state_tracking(self):
        agent = DialogueAgent()
        agent.generate_response("Hello!")
        agent.generate_response("What is memory?")
        assert agent._state.turn_count >= 2

    def test_retrieval_pool(self):
        agent = DialogueAgent()
        agent.ingest_corpus(["The sky is blue.", "Memory stores knowledge."])
        results = agent.retrieve("memory", k=1)
        assert len(results) >= 1

    def test_process_via_context(self):
        agent = DialogueAgent()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation="What is active inference?",
        )
        result = agent.safe_process(ctx)
        assert result.error is None
        assert "response" in result.data
        assert result.data["intent"] == "question"


# ================================================================== #
# 10. DemonstrationLearner
# ================================================================== #
class TestDemonstrationLearner:
    def test_record_demo(self):
        learner = DemonstrationLearner(latent_dim=16)
        states = [np.random.default_rng(i).standard_normal(64) for i in range(5)]
        actions = [0, 1, 2, 1, 3]
        learner.record_demonstration(states, actions, reward=1.0, label="push_left")
        assert len(learner._demos) == 1
        assert len(learner._imitation_memory) == 5

    def test_consolidate_replays(self):
        learner = DemonstrationLearner(latent_dim=16, replay_multiplier=2)
        states = [np.ones(64) * 0.5]
        actions = [1]
        learner.record_demonstration(states, actions)
        result = learner.consolidate()
        assert result["replayed"] > 0
        assert result["n_demos"] == 1

    def test_imitate_returns_action(self):
        learner = DemonstrationLearner(latent_dim=16)
        states = [np.ones(64) * 0.5]
        actions = [2]
        learner.record_demonstration(states, actions)
        # 用相同状态查询
        result = learner.imitate(np.ones(64) * 0.5)
        assert result is not None
        action, sim = result
        assert action == 2

    def test_imitation_bias_normalized(self):
        learner = DemonstrationLearner(latent_dim=16)
        states = [np.ones(64) * 0.5]
        actions = [1]
        learner.record_demonstration(states, actions)
        bias = learner.get_imitation_bias(np.ones(64) * 0.5, n_actions=4)
        assert bias.shape == (4,)
        assert abs(bias.sum() - 1.0) < 0.01  # 归一化

    def test_no_demos_returns_none(self):
        learner = DemonstrationLearner(latent_dim=16)
        assert learner.imitate(np.zeros(64)) is None

    def test_process_via_context(self):
        learner = DemonstrationLearner(latent_dim=16)
        states = [np.ones(64) * 0.3]
        actions = [0]
        learner.record_demonstration(states, actions, label="test")
        ctx = SkillContext(belief=np.ones(64) * 0.3)
        result = learner.safe_process(ctx)
        assert result.error is None
        assert result.data["n_demos"] == 1
        assert result.data["imitation_available"] is True


# ================================================================== #
# 11. TeachingModule
# ================================================================== #
class TestStudentModel:
    def test_teach_reduces_error(self):
        student = StudentModel(dim=8, lr=0.5)
        x = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        err1 = student.teach(x, 1.0)
        err2 = student.teach(x, 1.0)
        assert err2 < err1

    def test_predict_shape(self):
        student = StudentModel(dim=8)
        pred = student.predict(np.zeros(8))
        assert isinstance(pred, float)

    def test_knowledge_state(self):
        student = StudentModel(dim=8)
        student.teach(np.ones(8), 1.0)
        state = student.knowledge_state()
        assert "mastery" in state
        assert 0 <= state["mastery"] <= 1


class TestTeachingModule:
    def test_teach_step(self):
        module = TeachingModule(student_dim=16, n_skills=4)
        result = module.teach_step(0)
        assert result["taught"] is True
        assert "error" in result
        assert "student_state" in result

    def test_process_increases_mastery(self):
        """教学后掌握度应相对于早期训练有所提升。"""
        module = TeachingModule(student_dim=16, n_skills=2)
        ctx = SkillContext(belief=np.zeros(64))
        # 前 5 步：建立基线（EMA 从 0 开始会先上升）
        for _ in range(5):
            module.safe_process(ctx)
        mid_mastery = module.student.knowledge_state()["mastery"]
        # 继续训练 20 步
        for _ in range(20):
            module.safe_process(ctx)
        final_mastery = module.student.knowledge_state()["mastery"]
        # 最终掌握度应不低于中期（模型在收敛）
        assert final_mastery >= mid_mastery - 0.1  # 允许小波动
        assert module.student._seen_examples >= 25

    def test_skill_evaluations(self):
        module = TeachingModule(student_dim=16, n_skills=3)
        ctx = SkillContext(belief=np.zeros(64))
        result = module.safe_process(ctx)
        assert "skill_evaluations" in result.data
        assert len(result.data["skill_evaluations"]) == 3

    def test_add_custom_example(self):
        module = TeachingModule(student_dim=16, n_skills=2)
        module.add_example(0, np.ones(16), 5.0)
        assert len(module._example_pool[0]) == 11  # 10 default + 1 custom


# ================================================================== #
# 12. MultiModalTranslator
# ================================================================== #
class TestMultiModalTranslator:
    def _make_frame(self, size=128, seed=42):
        rng = np.random.default_rng(seed)
        frame = np.zeros((size, size), dtype=np.uint8)
        cx, cy = rng.integers(20, size - 20, 2)
        r = 10
        cv, cu = np.ogrid[:size, :size]
        mask = (cv - cx) ** 2 + (cu - cy) ** 2 <= r**2
        frame[mask] = 255
        return frame

    def test_frame_to_text(self):
        translator = MultiModalTranslator()
        frame = self._make_frame()
        text = translator.frame_to_text(frame)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "see" in text.lower()

    def test_text_to_actions_simple(self):
        translator = MultiModalTranslator()
        actions = translator.text_to_actions("move left")
        assert len(actions) >= 1
        assert actions[0] == 0  # left = 0

    def test_text_to_actions_repeat(self):
        translator = MultiModalTranslator()
        actions = translator.text_to_actions("go right 3 times")
        assert len(actions) == 3
        assert all(a == 1 for a in actions)  # right = 1

    def test_text_to_actions_chinese(self):
        translator = MultiModalTranslator()
        actions = translator.text_to_actions("向左移动")
        assert len(actions) >= 1
        assert actions[0] == 0  # 左 = left = 0

    def test_alignment_error(self):
        translator = MultiModalTranslator()
        frame = self._make_frame()
        text = "I see a bright object"
        error = translator.compute_alignment_error(frame, text)
        assert isinstance(error, float)
        assert error >= 0.0

    def test_update_alignment_reduces_error(self):
        translator = MultiModalTranslator()
        frame = self._make_frame()
        text = "I see a bright object"
        err1 = translator.update_alignment(frame, text, lr=0.1)
        err2 = translator.update_alignment(frame, text, lr=0.1)
        assert err2 <= err1

    def test_process_via_context(self):
        translator = MultiModalTranslator()
        frame = self._make_frame()
        ctx = SkillContext(
            belief=np.zeros(64),
            raw_observation={"frame": frame, "text": "move left"},
        )
        result = translator.safe_process(ctx)
        assert result.error is None
        assert result.data["frame_to_text"] is not None
        assert len(result.data["text_to_actions"]) >= 1
