# deepagents надстройка над langgraph
# просто агент работает только с tools
# deepagent - агент с plan, offload context, sub-agents + mcp
# курс deepagents from scratch от langchain (есть от гигачат тот же курс, но подустаревший)

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from datetime import date

from state import DeepAgentState
from tools import get_weather, write_todos, read_todos, ls, read_files, write_file

from research_tools import tavily_search, think_tool, get_today_str

from utils import format_messages
from subagents import create_task_tool

from prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)

load_dotenv()

llm = ChatOpenAI(
    # model='qwen/qwen3.5-9b',
    model='openai/gpt-oss-20b',
    # model='deepseek/deepseek-r1-0528-qwen3-8b',
    # base_url='http://10.215.88.70:1234/v1',
    base_url='http://localhost:1234/v1',
    api_key=SecretStr('fake'),
    temperature=.7,
)
INSTRUCTIONS = (
    "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTIONS
    + "\n\n" + "="*80 + "\n\n"
    + "# FILE SYSTEM USAGE\n"
    + FILE_USAGE_INSTRUCTIONS
    + "\n\n" + "="*80 + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=3,
        max_researcher_iterations=3,
        date=get_today_str(),
    )
    + f"\n\nТекущая дата (сегодня) {date.today().strftime('%d.%m.%y')}"
)

# agent = create_agent(
#     model=llm,
#     tools=[...],
#     state_schema=DeepAgentState,
#     system_prompt=INSTRUCTIONS,
# )

agent = create_agent(
    model=llm,
    tools=[
        get_weather,
        read_todos,
        write_todos,
        ls,
        read_files,
        write_file,
        # tavily_search,      # чтобы main мог сам иногда искать
        think_tool, 
        create_task_tool(
            tools=[get_weather, read_todos, write_todos, ls, read_files, write_file],
            subagents=[{
                "name": 'weather_agent',
                "description": 'Агент для получения погоды',
                "system_prompt": 'Ты агент для получения погоды. Используй инструменты и отвечай развернуто и понятно.',
                "tools": ['get_weather'],
            },
            {
                "name": "research-agent",
                "description": "Агент для веб-ресёрча по одной теме за раз",
                "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
                "tools": ["tavily_search", "think_tool"],
            }
            ],
            llm=llm,
            state_schema=DeepAgentState,
        )
    ],
    state_schema=DeepAgentState,
    system_prompt=INSTRUCTIONS
)

# message = "Какая погода в Самаре, так же посмотри через месяц. И сравни также с городом Нью-Йорк. Покажи в таблице. Сохрани в формате md в файле weather.md"

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Что такое Model Context Protocol"
        }
    ],
    "weather": 'Нет информации о погоде'
})
# После выполнения result содержит финальное состояние:
# result["messages"] — вся история
# result["todos"] — список задач (пустой, если всё выполнено)
# result["files"] — словарь с созданными файлами

print('-------')

format_messages(result["messages"])

print('-------')

print('todos', result['todos'])

print('-------')
print('files', result.get('files', ''))
