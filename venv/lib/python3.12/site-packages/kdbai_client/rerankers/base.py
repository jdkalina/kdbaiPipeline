from pandas import DataFrame
import numpy as np


class BaseReranker:
    def __init__(self, overfetch_factor: int = 2) -> None:
        self._overfetch_factor = overfetch_factor
        
    def rank(
        self,
        query: str,
        results: DataFrame,
        text_column: str,
        top_n: int
    ) -> DataFrame:
        """
        End-to-end reranking of documents.
        """
        pass

    @property
    def overfetch_factor(self):
        return self._overfetch_factor

    @overfetch_factor.setter
    def overfetch_factor(self, value):
        if not isinstance(value, (int, np.integer)):
            raise ValueError("overfetch_factor must be of type int")
        self._overfetch_factor = value