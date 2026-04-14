import os
from typing import Annotated, Any
from langchain.tools import tool

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain.tools import InjectedToolCallId

from state import DeepAgentState, Todo
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

from prompts import *


@tool
def get_weather(
    city: str,
    date: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str | Command[Any]:
    """Инструмент для получения погоды в указанном городе

    Args:
        city : str - город
        date: str - дата

    Returns:
        str - Погода в городе на дату
    """
    # print(state, tool_call_id)
    tavily_client = TavilyClient(api_key=os.getenv('TVLY_KEY', ''))
    response = tavily_client.search(
        query=f"Погода в городе {city} на дату {date}")
    weather = response['results'][0]['content']
    # print('result', weather)

    return Command(
        update={
            "weather": weather, # Кэшируем в состояние
            "messages": [ToolMessage(content=weather, tool_call_id=tool_call_id)] # Возвращает результат пользователю (через ToolMessage)
        }
    )
# Command — специальный объект, который говорит LangGraph: "Не просто верни результат, а обнови состояние и добавь сообщение"

@tool(description=WRITE_TODOS_DESCRIPTION)
def write_todos(
    tool_call_id: Annotated[str, InjectedToolCallId],
    todos: list[Todo]
) -> Command[tuple[()]]:
    """
    Create or update a list of todos.

    Args:
        todos: list[Todo] - A list of todo items.

    Returns:
        Command: A command object with an empty tuple as the update.

    """
    print("============ write todos ============")
    print(todos)
    print("============ write todos ============")

    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(content=f"Updates todos: {todos}", tool_call_id=tool_call_id)]
        }
    ) # Это прямая запись в состояние, минуя поток сообщений. В теории Deep Agents, инструмент планирования должен быть no-op (ничего не делает, только обновляет состояние). Здесь он обновляет todos в стейте.


@tool()
def read_todos(
    state: Annotated[DeepAgentState, InjectedState]
):
    """
    Read the list of todos.

    Returns:
        list[Todo]
    """
    print("============ read todos ============")
    print(state.get('todos', []))
    print("============ read todos ============")

    todos_string = "Current todos: " + \
        "\n".join(
            [f"{todo['content']} - {todo['status']}" for todo in state["todos"]])
    return todos_string # просто читает текущее состояние и возвращает строку для LLM

# этот инструмент даёт агенту осознание своего рабочего пространства. Без ls агент не знает, какие файлы уже есть.
@tool(description=LS_DESCRIPTION)
def ls(
    state: Annotated[DeepAgentState, InjectedState]
):
    """
    List of all files in the virtual file
    """
    return list(state.get('files', {}).keys()) # Возвращаем список имён файлов


@tool(description=READ_FILE_DESCRIPTION)
def read_files(
    state: Annotated[DeepAgentState, InjectedState],
    file_path: str,
    offset: int = 0,
    limit: int = 2000
) -> str:
    """
    Read content from a file in the file system with optional pagination.
    """
    file_content = state.get('files', {}).get(file_path, '')

    lines = file_content.split(sep='\n')

    return '\n'.join(lines[offset:offset+limit])


@tool(description=WRITE_FILE_DESCRIPTION)
def write_file(
    state: Annotated[DeepAgentState, InjectedState],
    file_path: str,
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[tuple[()]]:
    """
    Create or update a file in the file system.
    """
    files: dict[str, str] = state.get('files', {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [ToolMessage(content=f"Updates file: {file_path}", tool_call_id=tool_call_id)]
        }
    )


# InjectedState	Текущему состоянию агента
# InjectedToolCallId	ID вызова (для связки ответа с запросом)