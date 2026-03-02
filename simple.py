from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain.tools import tool

# llm = ChatOpenAI(
#     model='openai/gpt-oss-20b',
#     base_url='http://172.21.13.70:1234/v1',
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


answer = agent.invoke({
    "messages":[
{
    "role":"human",
    "content":"Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."
}
]
})

def format_message(message):
    """Форматирует одно сообщение для вывода (контент или вызов инструмента)."""
    if message.content:
        return message.content
    return f"{message.tool_calls[0]['name']}({message.tool_calls[0]['args']})"


print('---')
print(*[format_message(m) for m in answer['messages']], sep='\n----\n')
print('---')

# print(answer['messages'][-1].content)