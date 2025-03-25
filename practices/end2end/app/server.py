from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

import bs4

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

blog_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50)

splits = text_splitter.split_documents(blog_docs)

vectorstore = Chroma.from_documents(documents=splits,
                                    # embedding=CohereEmbeddings()
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()


# RAG-Fusion
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)


generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """

    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results


load_dotenv()

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0)

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI()

template2 = "Give me a summary about {topic} in a paragraph or less."
prompt2 = ChatPromptTemplate.from_template(template2)
chain2 = prompt2 | llm | StrOutputParser()

add_routes(app, chain2, path="/summary")
# add_routes(app, final_rag_chain, path="/query")

if __name__ == "__main__":
    import uvicorn
    import os
    
    host = os.environ["HOST"]
    port = os.environ["PORT"]

    uvicorn.run(app, host=host, port=port)
