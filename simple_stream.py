from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.tools import tool

# llm = ChatOpenAI(
#     model='openai/gpt-oss-20b',
#     base_url='http://10.13.135.70:1234/v1',
#     api_key=SecretStr('fake'),
#     temperature=.7,
# )

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

agent=create_agent(
    model=llm,
    tools=[get_price],
    system_prompt="Ты помощник по планированию покупок",
)


stream = agent.stream({
    "messages":[
{
    "role":"human",
    "content":"Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."
}
]
},

stream_mode=["messages", "updates"]
)



def format_message(message) -> str:
    """Форматирует одно сообщение для вывода (контент или вызов инструмента)."""
    if message.content:
        return message.content
    return f"{message.tool_calls[0]['name']}({message.tool_calls[0]['args']})"

step = 1

def format_chunk_message(chunk) -> str:
    """Форматирует одно сообщение для вывода (контент или вызов инструмента)."""
    message, meta = chunk
    global step
    langraph_step = meta.get("langraph_step")

    if langraph_step is not None and langraph_step != step:
        step = langraph_step
        print('\n --- ---- --- \n')

    if message.content:
        print(message.content, end = '', flush=False)

for chunk in stream:
    chunk_type, chunk_data = chunk

    if chunk_type=='messages':
        format_chunk_message(chunk_data)
    if chunk_type=='updates':
        if chunk_data.get('model', None):
            print('\n---')
            print(format_message(chunk_data['model']['messages'][-1]))
            print('---')
