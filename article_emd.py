
import os
import json

from typing import Any, Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


Embedding = list[float]

class Article:
    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return f"Article(text={self.text!r})"


@dataclass(frozen=True)
class LayerEmbeddings:
    """
    Data class to store the attention heads of a layer.
    """
    attention_heads: list[Embedding]

    @classmethod
    def from_dict(cls, emb_dict: dict):
        return cls(**emb_dict)


@dataclass(frozen=True)
class FullEmbeddings:
    """
    Data class to store the standard embedding as well as embeddings from the
    different attention heads for the various layers.
    """
    standard_embedding: Embedding
    layer_embeddings: dict[int, LayerEmbeddings]

    @classmethod
    def from_dict(cls, emb_dict: dict) -> 'FullEmbeddings':
        layer_embeddings: dict[int, LayerEmbeddings] = {}
        for layer_idx, l_emb_dict in emb_dict["layers"].items():
            layer_embeddings[int(layer_idx)] = LayerEmbeddings.from_dict(l_emb_dict)
        return cls(
            emb_dict["standard"],
            layer_embeddings
        )


@dataclass(frozen=True)
class ArticleEmbeddings:
    """
    Data class to store the article information as well as the full embeddings
    (standard embedding, attention head embeddings).
    """
    article: Article
    embeddings: FullEmbeddings

    @classmethod
    def from_dict(cls, emb_dict: dict) -> 'ArticleEmbeddings':
        return cls(
            Article.from_dict(emb_dict["article"]),
            FullEmbeddings.from_dict(emb_dict["embeddings"]),
        )


class EmbeddingModel:
    """
    Class that defines the interface for the Salesforce/SFR-Embedding-Mistral embedding model.
    """
    class CachingModule(torch.nn.Module):
        """
        Custom wrapper around an instance of :class:`nn.Module` that caches its inputs.
        """
        def __init__(self, module: torch.nn.Module) -> None:
            """
            Initialize the cache instance with the Torch module.

            :param module: Torch module.
            :type module: torch.nn.Module
            """
            super().__init__()
            self._module = module
            self.last_input: Optional[torch.Tensor] = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Store x and forward to wrapped :class:`nn.Module` instance.

            :param x: Input.
            :type x: torch.nn.Tensor
            :return: Output of the wrapped module class instance.
            :rtype: torch.nn.Tensor
            """
            self.last_input = x
            return self._module.forward(x)
        
    def __init__(self, target_layers: Optional[set[int]], device: str) -> None:
        """
        Initialize the embedding model.

        :param target_layers: Layers to target.
        :type target_layers: Optional[set[int]]
        :param device: The device to load the model on.
        :type device: str
        """
        self._tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        self._model = AutoModel.from_pretrained(
            "BAAI/bge-large-en-v1.5",  output_hidden_states=True, output_attentions=True
        ).to(device)
        self.device = device
        self.target_layers = target_layers or {len(self._model.encoder.layer) - 1}

        for layer in self._model.encoder.layer:
            layer.attention.output.dense = self.CachingModule(layer.attention.output.dense)

    def generate_embeddings(self, text: str) -> FullEmbeddings:
        """
        Generate embeddings (standard embedding, attention head embeddings) for the input text.

        :param text: Input text.
        :type text: str
        :return: Embeddings.
        :rtype: FullEmbeddings
        """
        all_layer_embeddings: dict[int, LayerEmbeddings] = {}

        def split(_embedding: Embedding, _interval: int) -> list[Embedding]:
            _sub_embeddings: list[Embedding] = []
            for i in range(0, len(_embedding), _interval):
                _sub_embeddings.append(_embedding[i:(i + _interval)])
            return _sub_embeddings

        with torch.no_grad():
            batch_dict = self._tokenizer([text], padding=True, return_tensors="pt", truncation=True)
            # batch_dict = self._tokenizer.prepare_for_model(inputs, return_tensors="÷pt", max_length=max_length, truncation=True)

            iids = batch_dict["input_ids"].to(self._model.device)
            attention_mask = batch_dict["attention_mask"].to(self.device)

            outputs = self._model(iids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states

            for layer_idx, layer in enumerate(self._model.encoder.layer):
                # run inference for this layer
                hidden_states = layer(hidden_states[0])
                if layer_idx in self.target_layers:
                    # retrieve cached input from custom module
                    attn_heads = layer.attention.output.dense.last_input
                    layer_embeddings = LayerEmbeddings(
                        attention_heads=split(attn_heads[0][-1].tolist(), 64),
                    )
                    all_layer_embeddings[layer_idx] = layer_embeddings

            # apply last layer, followed by the RMS norm
            standard_embedding = (hidden_states)[0][0][-1].tolist()
            return FullEmbeddings(standard_embedding, all_layer_embeddings)

def embed_articles(articles: list[Article], model: EmbeddingModel) -> list[ArticleEmbeddings]:
    """
    Embed Article documents.

    :param articles: List of Article documents.
    :type articles: list[Article]
    :param model: Embedding model to use.
    :type model: EmbeddingModel
    :return: List of embeddings for the Article documents.
    :rtype: list[ArticleEmbeddings]
    """
    for article in tqdm(articles, "Generating article embeddings"):
        embeddings = model.generate_embeddings(article.text)
        article_embedding = ArticleEmbeddings(article, embeddings)
        yield article_embedding


def generate_embeddings(
        # article_path: str,
        # query_path: str,
        articles,
        target_layers: Optional[set[int]],
        # export_path: str
) -> tuple[list[ArticleEmbeddings]]:
    """
    Generate embeddings of the documents in the dataset and the queries.

    :param article_path: Path to the JSON file containing the dataset.
    :type article_path: str
    :param query_path: Path to the JSON file containing the queries.
    :type query_path: str
    :param target_layers: Layers to target for the attention head embeddings.
    :type target_layers: Optional[set[int]]
    :param export_path: Path to the output file.
    :type export_path: str
    :return: Document and query embeddings.
    :rtype: tuple[list[ArticleEmbeddings], list[QueryEmbeddings]]
    """
    articles: list[Article] = articles

  
    article_embeddings = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingModel(target_layers, device)

    try:
        for article_emb in embed_articles(articles, model):
            article_embeddings.append(article_emb)

    except KeyboardInterrupt:
        print("Embedding generation canceled.")

    return article_embeddings