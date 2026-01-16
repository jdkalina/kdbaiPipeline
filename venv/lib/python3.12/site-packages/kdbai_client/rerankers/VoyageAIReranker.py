from .base import BaseReranker
from pandas import DataFrame, Series
import voyageai


class VoyageAIReranker(BaseReranker):
    def __init__(self, api_key: str, model: str = "rerank-2", overfetch_factor: int = 2):
        super().__init__(overfetch_factor=overfetch_factor)
        self.api_key = api_key
        self.model = model
        self.client = voyageai.Client(api_key=self.api_key)

    def rank(self, query: str, results: DataFrame, text_column: str, top_n: int):
        texts = results[text_column].to_list()
        response = self.client.rerank(
                model=self.model, 
                query=query, 
                documents=texts, 
                top_k=top_n
        )
        reranked_ids = [result.index for result in response.results]
        results = results.loc[reranked_ids]
        results = results.assign(relevance_score=Series([result.relevance_score for result in response.results], index=results.index).values)
        results = results.reset_index(drop=True)
        return results