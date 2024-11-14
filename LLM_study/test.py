from zhipuai_llm import ZhipuAILLM
from dotenv import find_dotenv, load_dotenv
import os
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

_ = load_dotenv(find_dotenv())
api_key = os.environ['ZHIPUAI_API_KEY']
zhipuai_model = ZhipuAILLM(model = "glm-4", temperature = 0.1, api_key = api_key)


embedding = ZhipuAIEmbeddings(api_key=api_key)

# 向量数据库持久化路径
persist_directory = './data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")
question="当代青年要坚定信念，投身新时代中国特色社会主义伟大事业如何做？"
docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(docs)}")

for i, doc in enumerate(docs):
    print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")


# template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
# 案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
# {context}
# 问题: {question}
# """
#
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
#                                  template=template)
#
# qa_chain = RetrievalQA.from_chain_type(zhipuai_model,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
# question_1 = "什么是南瓜书？"
# question_2 = "马克思主义是什么？"
#
# result = qa_chain({"query": question_1})
# print("大模型+知识库后回答 question_1 的结果：")
# print(result["result"])
#
# result = qa_chain({"query": question_2})
# print("大模型+知识库后回答 question_2 的结果：")
# print(result["result"])
#
# prompt_template = """请回答下列问题:
#                             {}""".format(question_1)
# zhipuai_model.predict(prompt_template)
#
# result = qa_chain({"query": question_2})
# print("大模型+知识库后回答 question_2 的结果：")
# print(result["result"])
#
# prompt_template = """请回答下列问题:
#                             {}""".format(question_1)
# zhipuai_model.predict(prompt_template)
#
# memory = ConversationBufferMemory(
#     memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
#     return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
# )
#
# retriever=vectordb.as_retriever()
#
# qa = ConversationalRetrievalChain.from_llm(
#     zhipuai_model,
#     retriever=retriever,
#     memory=memory
# )
# question = "我可以学习到关于马克思主义吗？"
# result = qa({"question": question})
# print(result['answer'])
#
# question = "为什么这门课需要教这方面的知识？"
# result = qa({"question": question})
# print(result['answer'])

