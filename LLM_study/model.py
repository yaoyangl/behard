import os
from dotenv import load_dotenv, find_dotenv
from zhipuai_llm import ZhipuAILLM

_ = load_dotenv(find_dotenv())

# 获取环境变量 ZHIPUAI_API_KEY
api_key = os.environ['ZHIPUAI_API_KEY']

