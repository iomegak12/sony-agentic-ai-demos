from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

import os


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    return embeddings


def search_similar_documents(query, no_of_documents, index_name, embeddings):
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    similar_documents = vector_store.similarity_search_with_score(
        query, k=no_of_documents)

    return similar_documents


def main():
    try:
        load_dotenv()

        index_name = os.environ["PINECONE_INDEX_NAME"]
        query = """
            Experienced Candidates with Embedded Systems
            
            Requirements:
            Bachelors Degree in Computer Science
            At least 5 Years of working experience in Embedded Systems
            Understanding of Computer Architecture, Programming Languages and Interfacing Technologies 
        """

        embeddings = create_embeddings()
        no_of_documents = 2

        relevant_documents = search_similar_documents(
            query, no_of_documents, index_name, embeddings)

        for doc_index in range(len(relevant_documents)):
            document, score = relevant_documents[doc_index]

            print(document.metadata["source"])
            print(document.page_content)
            print(score)
            print("\n\n\n")
    except Exception as error:
        print(f"Error Occurred, Details : {error}")

        raise error


if __name__ == "__main__":
    main()