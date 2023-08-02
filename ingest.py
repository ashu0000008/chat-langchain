"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader, PyPDFLoader, TextLoader, DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def ingest_docs():
    """Get documents from web pages."""
    # loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    # loader = PyPDFLoader("PAXPAY互联网终端接入规范V2.1.pdf")
    # loader = TextLoader("faqskill_FAQ_百度_UNIT.txt", autodetect_encoding=True)

    loader = DirectoryLoader('./test', glob='**/*.txt')
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("end")

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
