from __future__ import annotations

import logging
from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)

class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""

    api_key: str  # 添加 API 密钥属性
    client: Any  # 客户端对象

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """
        实例化ZhipuAI为values["client"]

        Args:
            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:
            values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        try:
            from zhipuai import ZhipuAI
            api_key = values.get('api_key')
            if not api_key:
                raise ValueError("API key is required")
            values["client"] = ZhipuAI(api_key=api_key)
        except ImportError as e:
            raise ModuleNotFoundError("Module 'zhipuai' not found, please install it with `pip install zhipuai`") from e
        return values

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            text (str): 要生成 embedding 的文本.

        Returns:
            List[float]: 输入文本的 embedding，一个浮点数值列表.
        """
        response = self.client.embeddings.create(
            model="embedding-2",
            input=text
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.

        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        return [self.embed_query(text) for text in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Official does not support asynchronous requests")