
# from langchain.agents import AgentState

# from langgraph.graph import add_messages
# from langchain_core.messages import AnyMessage
# from typing import Literal, NotRequired, Annotated, TypedDict


# def weather_reducer(prev: str, next: str) -> str:
#     return next if next else prev

# class Todo(TypedDict):
#     content:str
#     status: Literal["pending", "in_progress", "completed"]

# def file_reducer(prev: dict[str, str], next: dict[str, str]) -> dict[str, str]:
#     return{**prev, **next}

# class DeepAgentState(AgentState):
#     weather: Annotated[str, weather_reducer]
#     messages: Annotated[list[AnyMessage], add_messages]
#     todo: NotRequired[list[Todo]]
#     files: Annotated[NotRequired[dict[str,str]], file_reducer]

from typing import Annotated, Literal, NotRequired, TypedDict

from langgraph.graph import add_messages

from langchain_core.messages import AnyMessage

from langchain.agents import AgentState


def weather_reducer(prev: str, next: str) -> str:
    return next if next else prev


class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left: dict[str, str] | None, right: dict[str, str]) -> dict[str, str]:
    if left is None:
        return right

    if right is None:
        return left

    return {**left, **right}

# Когда несколько инструментов одновременно пишут в состояние, 
# reducer-ы определяют, как объединить изменения. Без них данные бы затирались.

class DeepAgentState(AgentState):
    messages: Annotated[list[AnyMessage], add_messages] # Обёртка, которая говорит LangGraph: "Тип поля — list[AnyMessage], и вот как их объединять — функцией add_messages"
    # add_messages: не заменяет список, а добавляет новые сообщения в конец. Без этого каждый новый вызов затирал бы предыдущие сообщения.
    weather: Annotated[str, weather_reducer]
    todos: NotRequired[list[Todo]] # NotRequired — поле может отсутствовать в начальном состоянии
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
