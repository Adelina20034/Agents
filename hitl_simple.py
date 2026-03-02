from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

console = Console()

llm = ChatOpenAI(
    model='openai/gpt-oss-20b',
    base_url='http://localhost:1234/v1',
    api_key=SecretStr('fake'),
    temperature=.7,
)


@tool
def get_price(city: str, product:str)->str:
    """Это инструмент для получения цен на продукты в указанном городе"""
    price_agent = create_agent(
        model =llm,
        system_prompt="""
        Требуется сгенерировать реалистичную цену на продукт в указанном городе, 
        опираясь на исторические данные о ценах. Субагент должен возвращать ответ в виде таблицы:
        | Продукт | Цена (руб.) | Магазин |
        """,
    )

    answer = price_agent.invoke({
    "messages":[
{
    "role":"human",
    "content":f"Цена на {product} в городе {city} (в руб.)"
}
]
})
    
    return answer['messages'][-1].content


memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_price],
    system_prompt='Ты помощник по планированию покупок',
    checkpointer=memory,
    interrupt_before=['tools']
)

config = { 'configurable': { 'thread_id': 'someThread' } }



def format_message(message) -> str:
    """Форматирует одно сообщение для вывода (контент или вызов инструмента)."""
    if message.content:
        return message.content
    return f"{message.tool_calls[0]['name']}({message.tool_calls[0]['args']})"

step = 1

def format_chunk_message(chunk):
    """Форматирует одно сообщение для вывода (контент или вызов инструмента)."""
    message, meta = chunk
    global step
    if meta["langgraph_step"] != step:
        step = meta["langgraph_step"]
        print("\n --- --- --- \n")
    if message.content:
        print(message.content, end="")


def ask_and_run(user_unput: str, config):
    for chunk in agent.stream(
        user_unput,
        config=config,
        stream_mode=["messages", "updates"],
    ):
        chunk_type, chunk_data = chunk
        if chunk_type == "messages":
            format_chunk_message(chunk_data)
        if chunk_type == "updates":
            if chunk_data.get("model", None):
                print(
                    format_message(chunk_data["model"]["messages"][-1]), sep="\n---\n"
                )

        state = agent.get_state(config)
        # if '__interrupt__' in state.next ==('tools', ):
        if state.next and state.next[0] == "tools":
            tool_call = state.values['messages'][-1].tool_calls[0]
            print(f'Агент хочет вызвать утилиту {tool_call['name']}({tool_call['args']})')

            answer = input('\nРазрешить? (Y/n): ')
            if answer.lower().strip() == 'y':
                ask_and_run(None, config)
            else:
                console.print('Отменено')
                break


while True:
    user_input = input("\nВы: ")
    if user_input == "exit":
        break

    ask_and_run({"messages": [{"role": "human", "content": user_input}]}, config)
    print('\n')
