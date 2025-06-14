import os
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores.supabase import SupabaseVectorStore
from supabase import create_client

app = FastAPI()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
vectorstore = SupabaseVectorStore(
    supabase_client,
    embeddings
)

from langchain.chains import ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, 
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

add_routes(app, qa_chain, path="/invoke")
