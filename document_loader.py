from langchain_community.document_loaders import DirectoryLoader

def load_documents():
    loader = DirectoryLoader('data/', glob="**/*.txt")
    docs = loader.load()
    return docs
