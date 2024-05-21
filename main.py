import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import chat_agent_executor
from langgraph.checkpoint.sqlite import SqliteSaver




os.environ["TAVILY_API_KEY"] = "TAVILY_API_KEY"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

search = TavilySearchResults(max_results=2)
# result = search.invoke("what is the weather in SF")

# Retriever
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

tools = [search, retriever_tool]

# Model
model = ChatOpenAI(model="gpt-4")
model_with_tools = model.bind_tools(tools)


#Agent
memory = SqliteSaver.from_conn_string(":memory:")
config = {"configurable": {"thread_id": "abc123"}}

agent_executor = chat_agent_executor.create_tool_calling_executor(
    model, tools, checkpointer=memory
)
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob!")]}, config
):
    print(chunk)
    print("----")

# Agent memory is saved in memory and can be loaded back in the future, for example:
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")
