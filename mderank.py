import re
from torch import Tensor
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# If you haven't downloaded NLTK stopwords, you can activate the line below:
import nltk
nltk.download("stopwords") #pyright: ignore[reportUnknownMemberType]


class MDERank:
    """
    MDERank class is used to extract keyphrases from a document using sentence-transformers
    with the model "avsolatorio/GIST-small-Embedding-v0". The process is as follows:
      1. Preprocess the text.
      2. Extract candidate keyphrases using spaCy, removing stopwords from each candidate.
      3. Encode the original text and the "masked" text (where candidates are replaced with “[MASK]”).
      4. Compute cosine similarity between the embeddings of the original text and the masked text.
         (Idea: if masking a candidate significantly changes the embedding, cosine similarity will decrease,
         indicating the candidate is important.)
      5. Rank candidates by similarity (lower similarity → more important candidate)
         and select the top_k candidates.
    """
    def __init__(self, model_name: str="avsolatorio/GIST-small-Embedding-v0"):
        # Load SentenceTransformer model and move it to the appropriate device
        self.model = SentenceTransformer(model_name)
        
        # Initialize spaCy with the English model (if not installed, run: python -m spacy download en_core_web_sm)
        self.spacy_nlp = spacy.load("en_core_web_sm")
        
        # Get English stopwords from NLTK
        self.stopwords = stopwords.words("english") # pyright: ignore[reportUnknownMemberType]
    
    def clean_text(self, text: str):
        """
        Remove special characters, tabs, and extra whitespace.
        """
        text = re.sub(r'[<>[\]{}]', ' ', text)
        text = text.replace("\t", " ")
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    def extract_candidates(self, document: str):
        """
        Use spaCy to extract noun_chunks (noun phrases) from the text,
        then remove tokens that are stopwords from each candidate.
        """
        doc_clean = self.clean_text(document)
        spacy_doc = self.spacy_nlp(doc_clean)
        candidates: list[str] = []
        for chunk in spacy_doc.noun_chunks:
            # Remove tokens if they are stopwords
            candidate_tokens = [token.text for token in chunk if token.text.lower() not in self.stopwords]
            candidate = " ".join(candidate_tokens).strip()
            if candidate:
                candidates.append(candidate)
        return candidates
    
    def generate_masked_document(self, document: str, candidate: str):
        """
        Replace the candidate in the text with the string "[MASK]" with the corresponding number of tokens.
        Simple method: use regex to find and replace.
        If the candidate is not found, return None.
        """
        # Use a simple method: work on the cleaned text and split by whitespace
        doc_clean = self.clean_text(document)
        # Split the candidate by whitespace
        candidate_tokens = candidate.split()
        if not candidate_tokens:
            return None
        # Create a mask string with the corresponding number of tokens
        mask_tokens = ["[MASK]"] * len(candidate_tokens)
        candidate_text = " ".join(candidate_tokens)
        # Use regex to replace the entire candidate (using pattern r"\b...\b")
        pattern = r"\b" + re.escape(candidate_text) + r"\b"
        if not re.search(pattern, doc_clean):
            return None
        masked_text = re.sub(pattern, " ".join(mask_tokens), doc_clean)
        return masked_text

    def encode_document(self, text: str):
        """
        Encode the text into an embedding tensor using sentence-transformers.
        """
        # Use model.encode with convert_to_tensor=True to get a tensor
        embedding: Tensor = self.model.encode(text, convert_to_tensor=True) # pyright: ignore[reportUnknownMemberType]
        return embedding

    def compute_similarity(self, ori_embedding: Tensor, masked_embedding: Tensor) -> float:
        """
        Compute cosine similarity between the embeddings of the original text and the masked text.
        """
        # Use the cosine_similarity function from sentence-transformers (in util)
        cosine_sim = util.cos_sim(ori_embedding, masked_embedding) # pyright: ignore[reportUnknownMemberType]
        # cosine_sim is a 1x1 tensor, return the scalar value
        return cosine_sim.item()

    def extract_keyphrases(self, document: str, top_k: int=10):
        """
        Main function: extract keyphrases from the document.
          1. Preprocess the text.
          2. Get the embedding of the original text.
          3. Extract candidate keyphrases.
          4. For each candidate, create a masked text and compute cosine similarity.
          5. Rank candidates by similarity (lower similarity → more important).
          6. Remove duplicates and return the top_k candidates.
        """
        doc_clean = self.clean_text(document)
        ori_embedding = self.encode_document(doc_clean)
        candidates = self.extract_candidates(doc_clean)
        
        scored_candidates: list[tuple[str, float]] = []
        for candidate in candidates:
            masked_doc = self.generate_masked_document(doc_clean, candidate)
            if masked_doc is None:
                continue
            masked_embedding = self.encode_document(masked_doc)
            score = self.compute_similarity(ori_embedding, masked_embedding)
            scored_candidates.append((candidate, score))
        
        # Sort candidates by increasing similarity (important candidates have lower similarity)
        scored_candidates = sorted(scored_candidates, key=lambda x: x[1])
        
        # Remove duplicates by converting to lowercase
        seen: set[str] = set()
        keyphrases: list[str] = []
        for cand, sim in scored_candidates:
            if cand.lower() in seen:
                continue
            seen.add(cand.lower())
            keyphrases.append(cand)
            if len(keyphrases) >= top_k:
                break
        return keyphrases


    


# Example usage:
# if __name__ == "__main__":
    
#     document = (
#         """The Cosmos ecosystem is set to transform blockchain interoperability with IBC Eureka, 
#         the upgrade to its Inter-Blockchain Communication (IBC) protocol. The first transaction 
#         between Cosmos Hub (ATOM) and Ethereum was successfully made on Friday, March 28, with 
#         potential implications for the user and developer experience in web3. 
#         Magnas Mareneck, co-CEO at the newly-formed Interchain Labs, commented: 
#         “EUREKA! Cosmos Hub MAINNET has now sent its first live IBC transaction to Ethereum… and back! 
#         The content: 1 $ATOM. The message: i love cosmos” 
#         Barry Plunkett, co-CEO of Interchain Labs, posted: 
#         “First ethereum, then the world. Over the rest of the year, we’re going to expand the newer, 
#         much simpler IBC protocol across all of crypto” 
#         IBC Eureka seeks to introduce significant improvements to IBC, including faster transactions, 
#         improved security, and expanded compatibility with non-Cosmos networks like Ethereum and Solana. 
#         IBC Eureka lowers technical barriers for developers by simplifying IBC implementation and optimizing 
#         packet relay processes. The first direct IBC transaction between Cosmos Hub and Ethereum demonstrates 
#         the protocol’s ability to connect independent blockchains natively, unlocking new possibilities for 
#         asset transfers and data sharing across diverse networks. The integration of Ethereum into Cosmos 
#         via IBC could significantly revitalize DeFi applications by enabling fluid movement of assets and 
#         creating sophisticated financial products. As Cosmos works to establish IBC as the universal standard 
#         for blockchain communication, the Eureka upgrade may bolster sentiment in the Cosmos ecosystem."""
#     )
    
#     async def main():
#         mde_rank = AsyncMDERank()
#         keyphrases = await mde_rank.extract_keyphrases(document, top_k=10)
#         print("Extracted keyphrases:", keyphrases)
    
#     asyncio.run(main())