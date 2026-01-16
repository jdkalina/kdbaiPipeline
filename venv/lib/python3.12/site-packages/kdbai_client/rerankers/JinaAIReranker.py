from .base import BaseReranker
from pandas import DataFrame, Series
import requests


class JinaAIReranker(BaseReranker):
    def __init__(self, api_key: str, model: str = "jina-reranker-v1-base-en", overfetch_factor: int = 2):
        super().__init__(overfetch_factor=overfetch_factor)
        self.api_key = api_key
        self.model = model

        self.endpoint = "https://api.jina.ai/v1/rerank"
        self.headers = {
                        "Content-type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }

    def rank(self, query: str, results: DataFrame, text_column: str, top_n: int):
        texts = results[text_column].to_list()
        data = {
            "model": self.model,
            "query": query,
            "top_n": top_n,
            "documents": texts
        }
        response = requests.post(url=self.endpoint, headers=self.headers, json=data)
        response_data = response.json()
        reranked_ids = [result['index'] for result in response_data['results']]
        results = results.loc[reranked_ids]
        results = results.assign(relevance_score=Series([result['relevance_score'] for result in response_data['results']], index=results.index).values)
        results = results.reset_index(drop=True)
        return results