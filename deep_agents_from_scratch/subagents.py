# # если не ук тулс то все тулс доступны
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, InjectedToolCallId, Tool, tool as langchain_tool_decorator
from typing import Annotated, Any, TypedDict
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from state import DeepAgentState
from prompts import TASK_DESCRIPTION_PREFIX

# class Subagent(TypedDict):
#     name: str
#     description: str
#     system_prompt: str
#     tools: list[str] | None


# def create_task_tool(
        
#     tools: list[BaseTool], 
#     subagents: list[Subagent], 
#     llm , 
#     state_schema) -> BaseTool:

#     subagent_registry:dict[Any, Any] = {}
#     tools_by_name: dict[str, Tool] = {}

        
#     for tool in tools:
#         tools_by_name[tool.name] = tool

#     for subagent in subagents:
#         if 'tools' in subagent and subagent['tools'] is not None: # если есть список tools → оставляешь ему только эти инструменты
#             subagent_tools: list[Tool] = [tools_by_name[tool_name] for tool_name in subagent['tools'] if tool_name in tools_by_name]
#         else: # иначе отдаёшь все tools
#             subagent_tools: list[Tool] = tools

#         subagent_registry[subagent['name']] = create_agent(
#             model=llm,
#             state_schema=state_schema,
#             system_prompt=subagent['system_prompt'],
#             tools=subagent_tools,

#         ) # создаёт отдельного агента‑граф с тем же state_schema, но своим system_prompt и своей наборкой tools

#     subagent_list_str: str = '\n'.join([f' - name: {subagent['name']}, description: {subagent['description']}' for subagent in subagents])
#     task_description: str = TASK_DESCRIPTION_PREFIX.format(other_agents=subagent_list_str)

#     @langchain_tool_decorator(description=task_description)
#     def task(
#             description: str, #текст задачи для саб‑агента
#             subagent_type: str, #имя саб‑агента
#             state: Annotated[DeepAgentState, InjectedState], #текущий state главного агента
#             tool_call_id: Annotated[str, InjectedToolCallId] #чтобы связать ToolMessage с этим вызовом
#              )-> Command[Any]:
        
#         if subagent_type not in subagent_registry:#Проверка, что такой саб‑агент есть
#             raise ValueError(f"Subagent {subagent_type} not found")

#         subagent: Any = subagent_registry[subagent_type]


#         subagent_state: dict[str, object] = dict(state) #Берёшь текущий state и делаешь копию
#         subagent_state['messages'] = [HumanMessage(content=description)]
        
#         subagent_result_state: Any = subagent.invoke(state)
#         subagent_last_message: Any = subagent_result_state['messages'][-1]
#         subagent_last_message_content: Any|str = getattr(subagent_last_message, 'content', '')
#         # result = 


#         return Command(
#         update={
#             "files": subagent_result_state.get('files', {}),
#             "messages": [ToolMessage(content=f"subagent last message: {subagent_last_message_content}", tool_call_id=tool_call_id)]
#         }
#     )

#     return task


# from langchain.messages import HumanMessage, ToolMessage
# from langchain.agents import create_agent
# from langchain_core.tools import BaseTool, InjectedToolCallId, Tool, tool as langchain_tool_decorator
# from typing import Annotated, Any, TypedDict
# from langgraph.types import Command

# from langgraph.prebuilt import InjectedState

# from deep_agents.state import DeepAgentState

# from deep_agents.prompts import TASK_DESCRIPTION_PREFIX


class SubAgent(TypedDict):
    name: str
    description: str
    system_prompt: str
    tools: list[str] | None


def create_task_tool(
        tools: list[Any],
        subagents,
        llm,
        state_schema: DeepAgentState) -> BaseTool:

    subagent_registry = {}
    tools_by_name: dict[str, Tool] = {}

    for tool in tools:
        tools_by_name[tool.name] = tool

    for subagent in subagents:
        subagent_tools: list[Tool]
        if 'tools' in subagent and subagent['tools'] is not None:
            subagent_tools = [tools_by_name[t_name]
                              for t_name in subagent['tools'] if t_name in tools_by_name]
        else:
            subagent_tools = tools

        subagent_registry[subagent['name']] = create_agent(
            model=llm,
            state_schema=state_schema,
            system_prompt=subagent['system_prompt'],
            tools=subagent_tools
        )

    subagents_list_str = '\n'.join(
        [f'- name: {subagent["name"]}, description: {subagent['description']}' for subagent in subagents])
    task_description = TASK_DESCRIPTION_PREFIX.format(
        other_agents=subagents_list_str)

    @langchain_tool_decorator(description=task_description)
    def task(
            description: str,
            subagent_type: str,
            state: Annotated[DeepAgentState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[Any]:

        if subagent_type not in subagent_registry:
            raise ValueError(f"Subagent {subagent_type} not found")

        subagent = subagent_registry[subagent_type]

        subagent_state = dict(state)
        subagent_state['messages'] = [HumanMessage(content=description)]

        subagent_result_state = subagent.invoke(state)
        subagent_last_message = subagent_result_state['messages'][-1]
        subagent_last_message_content = getattr(
            subagent_last_message, "content", '')

        return Command(
            update={
                "files": subagent_result_state.get('files', {}),
                "messages": [ToolMessage(content=f"subagent last message {subagent_last_message_content}", tool_call_id=tool_call_id)]
            }
        )

    return task
