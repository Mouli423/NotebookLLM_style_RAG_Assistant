import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_classic.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_community.document_loaders import YoutubeLoader
from langchain_classic.vectorstores import FAISS
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
hf_token=os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

llm = ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)


arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="search")
tavily_search=TavilySearch(max_results=5)

tools=[search,arxiv,wiki,tavily_search]

def document_loader(files):

    documents=[]
    for file in files:
        temp_pdf=f"./temp.pdf"
        with open(temp_pdf,"wb") as doc_file:
            doc_file.write(file.getvalue())
        loader=PyPDFLoader(temp_pdf)
        docs=loader.load()
        documents.extend(docs)
    splits=text_splitter.split_documents(documents)
    vector_store=FAISS.from_documents(documents=splits,embedding=embeddings)

    return vector_store


def youtube_data_loader(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    yt_data=loader.load()
    splits=text_splitter.split_documents(yt_data)
    vector_store=FAISS.from_documents(documents=splits,embedding=embeddings)

    return vector_store



def build_rag_chian(llm, retriever):
    

    contextualize_q_systemprompt=(
        " Use the chat history to answer the latest user question" \
        "which might related to context in the chat history," \
        "design a standalone question which can be understood without the " \
        "chat history. Do NOT answer the question, " \
        "just redesign it if needed and otherwise return is as it is."
        )

    contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_systemprompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]

        )
    
    history_aware_retriver=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    system_prompt = """
    You are a knowledgeable and honest AI assistant.

    Answer the user's question using the following priority order:

    1. If retrieved context is provided and relevant, use it.
    2. If context is incomplete or insufficient, you MAY use your own general knowledge to complete the answer.

    Rules:
    - Do NOT hallucinate facts.
    - If you use your own knowledge, clearly state that the answer is based on general knowledge.
    - If neither context nor general knowledge is sufficient, then say "I don't know".

    Retrieved context (if any):
    {context}
    """

    
    qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)
    return rag_chain

def get_session_history(session:str)->BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session]=ChatMessageHistory()
    return st.session_state.store[session]


   


st.title("📘 NotebookLLM-style RAG Assistant")

uploaded_files = st.sidebar.file_uploader("Choose PDF Files", type="pdf", accept_multiple_files=True,key="uploaded_files")

session_id = st.sidebar.text_input("Session ID", value="default_session")

url=st.sidebar.text_input("Youtube URL",key="url")

mode = st.sidebar.radio(
    "Select Input Mode",
    ["PDF RAG", "YouTube RAG", "Web / Wiki / Arxiv Search"],index=2,key="mode"
)
def chat_box(mode):
    return st.chat_input(
        f"Ask your question ({mode})",
        key=f"user_input_{mode}"
    )

user_input = chat_box(mode)

if "store" not in st.session_state:
    st.session_state.store = {}

if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = st.session_state.mode

def reset_inputs():
    for key in ["user_input","uploaded_files","url","vector_store","rag_chain"]:
        if key in st.session_state:
            del st.session_state[key]

if st.session_state.mode != st.session_state.prev_mode:
    reset_inputs()
    st.session_state.prev_mode = st.session_state.mode



if mode == "PDF RAG":

    url=None
    session_id="pdf123"
    if uploaded_files:
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = document_loader(uploaded_files)

        retriever = st.session_state.vector_store.as_retriever()


        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = build_rag_chian(llm, retriever)

        conversational_rag_chain = RunnableWithMessageHistory(
            st.session_state.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.chat_message("user").write(user_input)
            with st.chat_message("assistant"):
                st.write("Assistant:", response["answer"])

    else:
        st.info("Upload files to continue")


elif mode == "YouTube RAG":

    uploaded_files = None
    session_id="youtube123"

    if not url:
        st.info("Provide URL to continue")
        st.stop()


    if url:  

        if "vector_store" not in st.session_state:
            st.session_state.vector_store = youtube_data_loader(url)

        retriever = st.session_state.vector_store.as_retriever()

        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = build_rag_chian(llm, retriever)

        conversational_rag_chain = RunnableWithMessageHistory(
            st.session_state.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        if user_input:

            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.chat_message("user").write(user_input)
            with st.chat_message("assistant"):
                st.write(response["answer"])
        
elif mode == "Web / Wiki / Arxiv Search":

    uploaded_files=url=None

    
    if user_input:

        model=ChatGroq(model="qwen/qwen3-32b",api_key=groq_api_key)
        session_history = get_session_history(session_id)
        agent=create_agent(model=model,tools=tools,
                        system_prompt = """
                                You are a research assistant.

                                When answering:
                                - Use tools (Wikipedia, Arxiv, Web, tavily) for factual or external information
                                - You MAY answer directly for conceptual or well-known explanations
                                - Prefer tools when accuracy or freshness matters
                                - If tool results are empty, say "I could not find relevant information"

                                Maintain conversation context when answering follow-up questions.
                                                                """,
        )
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
            )
            st.write(response["messages"][-1].content)



else:
    st.warning("upload files or enter youtube url then ask your question")

