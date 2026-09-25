"""
Root alias for Global Judge Model Configuration.
Exposes ResilientNemotronJudge and get_judge_model from evals.llm_as_judge.
Primary Model: gemini-3.5-flash-lite
Secondary Model: nvidia/nemotron-3.5-lightning:free
"""

from evals.llm_as_judge import ResilientNemotronJudge, get_judge_model

__all__ = ["ResilientNemotronJudge", "get_judge_model"]

if __name__ == "__main__":
    judge = get_judge_model()
    print("Testing Root llm_as_judge alias:")
    print("Target:", judge.get_model_name())
    print(judge.generate("Respond with 'Judge Online'"))
