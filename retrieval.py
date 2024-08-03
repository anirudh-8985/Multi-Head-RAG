import heapq

from abc import abstractmethod
from typing import Callable
from operator import itemgetter

from article_emd import (
    Article,
    FullEmbeddings
)

from query_embed import (
    QueryEmbeddings
)

from db_store import (
    VectorDB
)

class MultiHeadStrategy(Strategy):
    """
    The MultiHeadStrategy class uses the Multi-Head RAG strategy for the evaluation.

    Inherits from the Strategy class and implements its abstract methods.
    """

    def __init__(
            self,
            name: str,
            db: VectorDB,
            layer: int,
            weight_fn: Callable[[float, int, float], float],
    ) -> None:
        """
        Initialize the MultiHeadStrategy instance with a name, a vector database instance,
        layer information as well as a vote function.

        :param name: Name of the strategy class.
        :type name: str
        :param db: Vector database instance.
        :type db: VectorDB
        :param layer: Layer to use embeddings from.
        :type layer: int
        :param weight_fn: Function to compute votes for a document based on head-scale,
            rank, and distance between query and document.
        :type weight_fn: Callable[[float, int, float], float]
        """
        super().__init__(name, db)
        self.weight_fn = weight_fn
        self.layer = layer

    def _search(self, emb: FullEmbeddings, n: int) -> list[list[tuple[float, Article]]]:
        """
        Search for closet neighbors of emb within the space of each attention head.

        :param emb: Query embeddings.
        :type emb: FullEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: List with search results (ordered list of (distance, Article) pairs)
            for each attention head.
        :rtype: list[list[tuple[float, Article]]]
        """
        return self.db.attention_search(emb, self.layer, n)

    def _get_head_scales(self) -> list[float]:
        """
        Get the scales for each attention head. The scale of an attention head is the
        product of the mean pairwise distance between documents for that head, and the mean
        embedding norm of all documents of that head.

        :return: List with the attention scales.
        :rtype: list[float]
        """
        return self.db.attention_scales

    def _multi_vote(self, emb: FullEmbeddings, n: int) -> list[tuple[float, Article]]:
        """
        Accumulate all votes over all attention heads. Each head votes for its n closest
        documents for the provided embedding, with the i-th closest receiving 2**-i votes.
        All votes are scaled with the respective head's head-scale.

        :param emb: Query embedding to retrieve documents for.
        :type emb: FullEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: Sorted list of the top n (votes, Article) pairs (most to least votes).
        :rtype: list[tuple[float, Article]]
        """
        votes: dict[Article, float] = {}
        ranking = self._search(emb, n)
        head_scales: list[float] = self._get_head_scales()

        for i, head in enumerate(ranking):
            for rank, (dist, voted) in enumerate(head[:n]):
                votes[voted] = votes.get(voted, 0.0) + self.weight_fn(head_scales[i], rank, dist)

        top_picks: list[tuple[Article, float]] = heapq.nlargest(n, votes.items(), key=itemgetter(1))
        return [(votes, article) for (article, votes) in top_picks]

    def _get_picks(self, query_embs: QueryEmbeddings, n: int) -> tuple[Article, ...]:
        """
        Use _multi_vote to pick the top n documents to retrieve, return the documents
        in order from the first to the nth pick.

        :param query_embs: Query embeddings to evaluate.
        :type query_embs: QueryEmbeddings
        :param n: Number of documents to retrieve.
        :type n: int
        :return: List of the n retrieved documents.
        :type: tuple[Article, ...]
        """
        return tuple(doc for (votes, doc) in self._multi_vote(query_embs.embeddings, n))
    
def weight_function(head_scale: float, rank: int, distance: float) -> float:
    return head_scale * (2 ** -rank) * distance