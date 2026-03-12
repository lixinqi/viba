import os

from openai import OpenAI


def _call_claude(prompt: str) -> str:
    """
    Call the LLM via SiliconFlow's OpenAI-compatible API.
    Requires SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, and SILICONFLOW_MODEL
    environment variables.
    """
    env = os.environ
    client = OpenAI(
        api_key=env.get('SILICONFLOW_API_KEY'),
        base_url=env.get('SILICONFLOW_BASE_URL'),
    )
    response = client.chat.completions.create(
        model=env.get('SILICONFLOW_MODEL'),
        messages=[
            {'role': 'user', 'content': prompt},
        ],
    )
    return response.choices[0].message.content.strip()
