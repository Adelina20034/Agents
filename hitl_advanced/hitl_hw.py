from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.7,
)

@tool
def get_weather(city: str, date: str) -> str:
    """Это инструмент для получения погоды в указанном городе"""
    weather_agent = create_agent(
        model=llm,
        system_prompt="""
Требуется предоставить прогнооз погоды для указанного города в виде таблицы
|Дата|Температура|Ветер|Давление|
на указанную дату. Если данных нет сформируй реалистичный ответ, заполни все ячейки таблицы.
Сделай прогноз на основании исторических тенденций
""",
    )
    answer = weather_agent.invoke(
        {
            "messages": [
                {
                    "role": "human",
                    "content": f"Какая погода в городе {city} на дату {date}?",
                }
            ]
        }
    )
    return answer["messages"][-1].content


memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt='Ты полезный ассистент',
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_weather": True,  # все решения: approve, edit, reject
                # "get_weather": {"allowed_decisions": ["approve", "reject"]},  # без edit
            },
            description_prefix="Подтвердите вызов инструмента...",
        ),
    ],
    checkpointer=memory,
)

config = {"configurable": {"thread_id": "someThread"}}


def ask_and_run(user_input: str, config):
    
    result = agent.invoke(
    {"messages": [{"role": "human", "content": user_input}]},
    config=config,
    )

    while '__interrupt__' in result:
        interrupt_value = result['__interrupt__'][0].value
        action_requests = interrupt_value['action_requests']
        review_configs = interrupt_value['review_configs']

        decisions = []


        for i, action in enumerate(action_requests):
            name = action["name"]
            args = action["args"]
            description = action.get("description", "")
            allowed = review_configs[i]["allowed_decisions"]

            print("\n--- Подтверждение ---")
            # print(f"Инструмент: {name}")
            # print(f"Аргументы: {args}")
            if description:
                print(f"{description}")
 
            print(f"Разрешённые решения: {allowed}")


            while True:
                decision_input = input("a = approve, r = reject: ").strip().lower()
                if decision_input == "a" and "approve" in allowed:
                    decisions.append({"type": "approve"})
                    break
                elif decision_input == "r" and "reject" in allowed:
                    reason = input("Сообщение для агента (причина отказа): ")
                    decisions.append({"type": "reject", "message": reason})
                    break
                else:
                    print("Некорректный ввод, попробуй ещё раз.")
                
        result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
        )
        
    final_message = result["messages"][-1].content
    print("\nАгент:", final_message)
    return result
    


while True:
    user_input = input("\nВы: ")
    if user_input == "exit":
        break

    ask_and_run(user_input, config)
    print('\n')
