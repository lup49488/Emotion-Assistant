from __future__ import annotations

import os

from openai import OpenAI


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY，再运行这个手动连通性测试。")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "您好"},
        ],
        stream=False,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
