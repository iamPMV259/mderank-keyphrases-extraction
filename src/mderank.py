import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk import pos_tag, word_tokenize
import asyncio
from scipy.stats import zscore

# Download necessary data for NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class MDERank:
    def __init__(self, model_name="bert-base-uncased", pooling="max"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def compute_embedding(self, text):
        # Normalize input and get output from the BERT model
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get hidden_states: [batch_size, sequence_length, hidden_size]
        hidden_states = outputs.last_hidden_state[0]  # (seq_len, hidden_size)
        if self.pooling == "max":
            embedding, _ = torch.max(hidden_states, dim=0)
        elif self.pooling == "avg":
            embedding = torch.mean(hidden_states, dim=0)
        else:
            embedding = torch.mean(hidden_states, dim=0)
        return embedding.numpy()

    def extract_candidates(self, text):
        """
        Use NLTK to tokenize, assign POS tags, and extract candidate phrases
        based on the pattern: consecutive words with tags starting with JJ (adjective) or NN (noun).
        """
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        candidates = []
        candidate = []
        for word, tag in tagged:
            if tag.startswith("JJ") or tag.startswith("NN"):
                candidate.append(word)
            else:
                if candidate:
                    phrase = " ".join(candidate)
                    candidates.append(phrase)
                    candidate = []
        if candidate:
            phrase = " ".join(candidate)
            candidates.append(phrase)
        # Remove duplicate phrases and keep phrases with at least 1 word
        candidates = list(set([c for c in candidates if len(c.split()) >= 1]))
        return candidates

    def mask_text(self, text, candidate):
        """
        Replace occurrences of the candidate in the text with [MASK] tokens corresponding to the number of tokens.
        """
        candidate_tokens = candidate.split()
        mask_token = " ".join(["[MASK]"] * len(candidate_tokens))
        # Use regex for replacement, case-insensitive
        pattern = re.compile(re.escape(candidate), re.IGNORECASE)
        masked_text = pattern.sub(mask_token, text)
        return masked_text

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

    def rank_keyphrases(self, text):
        """
        Compute the embedding of the original text and for each candidate, compute the embedding of the masked text.
        Then, calculate the cosine similarity between these embeddings. Candidates with lower similarity (indicating
        higher information loss) are considered more important.
        """
        original_embedding = self.compute_embedding(text)
        candidates = self.extract_candidates(text)
        scores = {}
        for candidate in candidates:
            masked_text = self.mask_text(text, candidate)
            masked_embedding = self.compute_embedding(masked_text)
            sim = self.cosine_similarity(original_embedding, masked_embedding)
            scores[candidate] = sim
        # Sort candidates in ascending order of similarity

        results = [(k, v) for k, v in scores.items()]
        keyphrases = [kw[0] for kw in results]
        scores = [kw[1] for kw in results]
        z_score = zscore(scores)
        selected_keyphrases = [(kw, z) for kw, z in zip(keyphrases, z_score) if z <= 0.5]

        # Sort candidates by ascending cosine similarity (candidates with higher information loss have lower similarity values)
        ranked = sorted(selected_keyphrases, key=lambda x: x[1])
        return ranked


class AsyncMDERank:
    def __init__(self, model_name="bert-base-uncased", pooling="max"):
        """
        Initialize the AsyncMDERank class with a BERT model and pooling method.

        Args:
            model_name (str): Name of the BERT model.
            pooling (str): Pooling method, default is "max". Other options can be "avg".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    async def compute_embedding(self, text):
        """
        Compute the embedding of the text using the BERT model asynchronously.

        Args:
            text (str): Input text.

        Returns:
            numpy.ndarray: Embedding of the text as a numpy array.
        """
        def _compute():
            # Normalize input and get output from the BERT model
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Get hidden_states: [seq_len, hidden_size]
            hidden_states = outputs.last_hidden_state[0]
            if self.pooling == "max":
                embedding, _ = torch.max(hidden_states, dim=0)
            elif self.pooling == "avg":
                embedding = torch.mean(hidden_states, dim=0)
            else:
                embedding = torch.mean(hidden_states, dim=0)
            return embedding.numpy()
        
        return await asyncio.to_thread(_compute)

    async def extract_candidates(self, text):
        """
        Use NLTK to tokenize, assign POS tags, and extract candidate phrases based on the pattern:
        consecutive words with tags starting with 'JJ' (adjective) or 'NN' (noun).

        Args:
            text (str): Input text.

        Returns:
            list[str]: List of candidate phrases (duplicates removed).
        """
        def _extract():
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            candidates = []
            candidate = []
            for word, tag in tagged:
                if tag.startswith("JJ") or tag.startswith("NN"):
                    candidate.append(word)
                else:
                    if candidate:
                        phrase = " ".join(candidate)
                        candidates.append(phrase)
                        candidate = []
            if candidate:
                phrase = " ".join(candidate)
                candidates.append(phrase)
            # Remove duplicate phrases and keep only phrases with at least 1 word
            candidates = list(set([c for c in candidates if len(c.split()) >= 1]))
            return candidates
        
        return await asyncio.to_thread(_extract)

    def mask_text(self, text, candidate):
        """
        Replace all occurrences of the candidate in the text with [MASK] tokens corresponding to the number of tokens.

        Args:
            text (str): Original text.
            candidate (str): Phrase to mask.

        Returns:
            str: Text after masking.
        """
        candidate_tokens = candidate.split()
        mask_token = " ".join(["[MASK]"] * len(candidate_tokens))
        pattern = re.compile(re.escape(candidate), re.IGNORECASE)
        masked_text = pattern.sub(mask_token, text)
        return masked_text

    def cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1 (numpy.ndarray): First vector.
            vec2 (numpy.ndarray): Second vector.

        Returns:
            float: Cosine similarity value.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

    async def rank_keyphrases(self, text):
        """
        Compute the embedding of the original text and for each candidate, compute the embedding of the masked text.
        Then, calculate the cosine similarity between these embeddings. Candidates with lower similarity (indicating
        higher information loss) are considered more important.

        Args:
            text (str): Input text.

        Returns:
            list[tuple[str, float]]: List of (candidate, similarity) pairs sorted in ascending order of similarity.
        """
        # Compute embedding for the original text and extract candidates asynchronously
        original_embedding = await self.compute_embedding(text)
        candidates = await self.extract_candidates(text)

        async def process_candidate(candidate):
            masked_text = self.mask_text(text, candidate)
            masked_embedding = await self.compute_embedding(masked_text)
            sim = self.cosine_similarity(original_embedding, masked_embedding)
            return candidate, sim

        # Run computations concurrently for all candidates
        tasks = [process_candidate(candidate) for candidate in candidates]
        results = await asyncio.gather(*tasks)
        keyphrases = [kw[0] for kw in results]
        scores = [kw[1] for kw in results]
        z_score = zscore(scores)
        selected_keyphrases = [(kw, z) for kw, z in zip(keyphrases, z_score) if z <= 0.5]

        # Sort candidates by ascending cosine similarity (candidates with higher information loss have lower similarity values)
        ranked = sorted(selected_keyphrases, key=lambda x: x[1])

        return ranked
