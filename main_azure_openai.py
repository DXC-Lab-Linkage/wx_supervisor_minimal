import os
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool

SUPERVISOR_STATE: Dict[str, Any] = {}

@tool
def set_state(prepared: bool) -> str:
    """Set 'prepared' in the supervisor-like state."""
    SUPERVISOR_STATE['prepared'] = bool(prepared)
    return f"state.updated: prepared={SUPERVISOR_STATE['prepared']}"

def load_prompt(name: str) -> str:
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "prompts", f"{name}.md"), encoding="utf-8") as f:
        return f.read().strip()

def make_llm() -> AzureChatOpenAI:
    # ★ watsonx -> AzureChatOpenAI に最小限の置換
    # 必要な環境変数:
    #   AZURE_OPENAI_API_KEY
    #   AZURE_OPENAI_ENDPOINT
    #   AZURE_OPENAI_API_VERSION   (例: "2024-02-15-preview")
    #   AZURE_OPENAI_DEPLOYMENT    (Azure 側で作成したデプロイ名)
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0,
        max_tokens=256,
    )

def build_graph():
    llm = make_llm()

    agent_a = create_react_agent(
        model=llm,
        tools=[set_state],
        prompt=load_prompt("agent_a_prompt"),
        name="agent_a",
    )
    agent_b = create_react_agent(
        model=llm,
        tools=[],
        prompt=load_prompt("agent_b_prompt"),
        name="agent_b",
    )

    supervisor = create_supervisor(
        model=llm,
        agents=[agent_a, agent_b],
        prompt=load_prompt("supervisor_system_prompt"),
        add_handoff_back_messages=True,   # ★戻りハンドオフを許可（多段ハンドオフを再現）
        output_mode="full_history",       # ★多段の様子を観測しやすく
    ).compile()
    return supervisor

def run_demo():
    load_dotenv()
    supervisor = build_graph()

    user_msg = "準備してから仕上げまで一気に実施してください。"
    print("\n=== SINGLE TURN (expect: Supervisor -> Agent A -> Supervisor -> Agent B) ===")
    events = supervisor.stream({"messages": [("user", user_msg)]}, stream_mode="values")
    for ev in events:
        last = ev.get("messages", [])[-1]
        print(f"[{ev.get('current_agent','?')}] {getattr(last, 'type','?')} -> {getattr(last, 'content', last)}")
    print("STATE:", SUPERVISOR_STATE)

if __name__ == "__main__":
    run_demo()
