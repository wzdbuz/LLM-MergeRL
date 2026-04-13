# 调用DeepSeek API
import json
import os
from openai import OpenAI
from llm_module.prompt import build_merge_prompt
from llm_module.semantic_prior import SemanticPrior


class LLMClient:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-chat"
        self.call_count = 0

    def get_prior(self, obs) -> SemanticPrior:
        prompt = build_merge_prompt(obs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,    # 低温度保证输出稳定
            max_tokens=200,
        )

        self.call_count += 1
        raw = response.choices[0].message.content.strip()

        # 解析 JSON
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        return SemanticPrior(
            risk_level=float(data["risk_level"]),
            merge_urgency=float(data["merge_urgency"]),
            gap_adequacy=float(data["gap_adequacy"]),
            speed_advice=float(data["speed_advice"]),
        )