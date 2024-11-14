import os
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from dotenv import load_dotenv,find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
_ = load_dotenv(find_dotenv())
file_path = ["./knowledge_db/mayuan.pdf"]
api_key = os.environ['ZHIPUAI_API_KEY']
def data_loader(file_path):
    loaders = []
    for file_path in file_path:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            loaders.append(PyMuPDFLoader(file_path))
        elif file_type == 'md':
            loaders.append(UnstructuredMarkdownLoader(file_path))
    return loaders

texts = []
loaders = data_loader(file_path)
for loader in loaders: texts.extend(loader.load())

text = texts[1]
print(f"每一个元素的类型：{type(text)}.",
    f"该文档的描述性数据：{text.metadata}",
    f"查看该文档的内容:\n{text.page_content[0:]}",
    sep="\n------\n")

# 切分文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)


embedding = ZhipuAIEmbeddings(api_key=api_key)
persist_directory = './data_base/vector_db/chroma'
vectordb = Chroma.from_documents(
    documents=split_docs[:100],
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

vectordb.persist()
print(f"向量库中存储的数量：{vectordb._collection.count()}")
