import uuid
from typing import Optional
from typing_extensions import TypedDict

import questionary

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

class State(TypedDict):
    """The graph state."""

    foo: str
    human_value: Optional[str]
    """Human value will be updated using an interrupt."""

def node(state: State):
    user_answer = interrupt(
        {
            "type": "alert",
            "question": "Уверены что хотите продолжить?",
            "allow_responds": ["approve", "reject"]
        }
    )
    print(user_answer)
    print(f"> Received an input from the interrupt: {user_answer['answer']}")
    return {"human_value": user_answer['answer']}

builder = StateGraph(State)
builder.add_node("node", node)
builder.add_edge(START, "node")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}

for chunk in graph.stream({"foo": "abc"}, config):
    # print(chunk)
    if '__interrupt__' in chunk:
        interrupt_value = chunk.get('__interrupt__', [{ "value": None }])[0].value
        print('Произошла остановка')
        print(interrupt_value)

        print(f"!!! {interrupt_value.get('type')} !!!")

        if 'allow_responds' in interrupt_value:
            user_input = questionary.select(
                interrupt_value.get('question'),
                choices=interrupt_value.get('allow_responds')
            ).ask()

            interrupt_value['answer'] = user_input

            command = Command(resume=interrupt_value)

            for chunk in graph.stream(command, config):
                print(chunk)
