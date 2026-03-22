import uuid
from typing import Optional
from typing_extensions import TypedDict

import questionary
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command




class StoryState(TypedDict):
    topic: str                 # тема / запрос пользователя
    setup: Optional[str]       # завязка истории
    options: Optional[list[str]]  # варианты действий героя
    choice: Optional[str]      # выбранный пользователем вариант
    ending: Optional[str]      # концовка истории



llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.7,
)

def story_node(state: StoryState) -> StoryState:
    topic = state["topic"]


    if not state.get("setup"):
        prompt = f"""
Тема: {topic}.
Придумай короткую завязку (2 предложения) и ровно 3 КОРОТКИХ варианта поступка героя.
Ответь строго в таком формате:

ЗАВЯЗКА:
<текст завязки в 1–3 предложениях>

ВАРИАНТЫ:
1) <вариант 1>
2) <вариант 2>
3) <вариант 3>
""".strip()

        resp = llm.invoke(prompt)
        text = resp.content

        setup = ""
        options: list[str] = []
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        mode = None
        for line in lines:
            if line.upper().startswith("ЗАВЯЗКА"):
                mode = "setup"
                continue
            if line.upper().startswith("ВАРИАНТЫ"):
                mode = "options"
                continue
            if mode == "setup":
                setup += (" " + line) if setup else line
            elif mode == "options":
                if ")" in line:
                    _, opt = line.split(")", 1)
                    options.append(opt.strip())
                else:
                    options.append(line)

        state["setup"] = setup
        state["options"] = options

        payload = interrupt(
            {
                "type": "choice",
                "question": f"{setup}\n\nЧто делаем?",
                "options": options,
            }
        )


        user_choice = payload["answer"]
        state["choice"] = user_choice

    if state.get("choice") and not state.get("ending"):
        setup = state["setup"]
        options = state["options"]
        user_choice = state["choice"]

        ending_prompt = f"""
Завязка истории:
{setup}

Варианты действий героя:
{chr(10).join(f"- {opt}" for opt in options)}

Пользователь выбрал:
{user_choice}

Напиши короткую концовку (2–3 предложения), логично продолжающую историю с учётом этого выбора.

""".strip()

        ending_resp = llm.invoke(ending_prompt)
        state["ending"] = ending_resp.content

    return state

builder = StateGraph(StoryState)
builder.add_node("story", story_node)
builder.add_edge(START, "story")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
    }
}


def run_game(topic: str):

    for chunk in graph.stream({"topic": topic, "setup": None,
                               "options": None, "choice": None,
                               "ending": None}, config):
        if "__interrupt__" in chunk:
            interrupt_obj = chunk["__interrupt__"][0].value
            question = interrupt_obj["question"]
            options = interrupt_obj["options"]


            answer = questionary.select(
                question,
                choices=options,
            ).ask()


            interrupt_obj["answer"] = answer

            for out_chunk in graph.stream(Command(resume=interrupt_obj), config):

                if "story" in out_chunk:
                    state = out_chunk["story"]
                    # print("\n=== ИТОГОВАЯ ИСТОРИЯ ===\n")
                    # print("Тема:", state["topic"])
                    # print("\nЗавязка:\n", state["setup"])
                    # print("\nВыбор пользователя:\n", state["choice"])
                    print("[LLM]  ", state["ending"])
            break


if __name__ == "__main__":
    topic = input("Тема истории: ")
    run_game(topic)
