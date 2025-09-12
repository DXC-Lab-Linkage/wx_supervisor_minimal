## 概要

watsonx と Azure OpenAI の LangGraph の supervisor の挙動を検証するソースコードです。
watsonx の不具合は 2025/09/12 時点で確認しています。それ以降に修正されている場合がございます。

## 構成

```
wx_supervisor_minimal/
├─ main.py
├─ requirements.txt
├─ .env.example
└─ prompts/
   ├─ supervisor_system_prompt.md
   ├─ agent_a_prompt.md
   └─ agent_b_prompt.md
```

## 使い方

仮想環境を作成して依存を入れる

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

.env.example を .env にコピーして、watsonx の各種値を設定

```
cp .env.example .env
# WATSONX_URL / WATSONX_APIKEY / WATSONX_PROJECT_ID / WATSONX_MODEL_ID
```

をセット

## 実行

```
python main.py
```
