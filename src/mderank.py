import re
import torch
import nltk
from nltk.corpus import stopwords
import spacy
import torch.nn.functional as F
import asyncio
from transformers import AutoTokenizer, AutoModel


# Ensure NLTK resources are downloaded (if not, uncomment the lines below)
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

class MDERank:
    """
    
    Main steps:
      1. Preprocess the document.
      2. Extract candidate keyphrases using spaCy (based on noun_chunks).
      3. For each candidate, replace (mask) the candidate in the document and encode it as input for the model.
      4. Compute cosine similarity between the embedding of the original document and the masked document.
      5. Rank the candidates by score (lower scores indicate higher importance of the candidate).
    """
    def __init__(self, model_name="avsolatorio/GIST-small-Embedding-v0", 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 max_len=512):
        self.device = device
        self.max_len = max_len

        # Load tokenizer and model from transformers (using AutoModel)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize spaCy pipeline (using English model)
        self.spacy_nlp = spacy.load("en_core_web_sm")
        
        # Load English stopwords
        self.stopwords = set(stopwords.words('english'))
        
        # (If needed) We still keep NLTK's RegexpParser to handle special cases.
        self.grammar = "NP: {<NN.*|JJ>*<NN.*>}"
        self.np_parser = nltk.RegexpParser(self.grammar)

    def clean_text(self, text):
        """
        Preprocess text: remove special characters, replace tabs, and remove extra whitespace.
        """
        text = re.sub(r'[<>[\]{}]', ' ', text)
        text = text.replace("\t", " ")
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def extract_candidates(self, document):
        """
        Extract candidate keyphrases using spaCy:
          - Use spaCy to tokenize and POS-tag the text.
          - Extract noun_chunks (noun phrases) from the text.
        """
        doc_clean = self.clean_text(document)
        spacy_doc = self.spacy_nlp(doc_clean)
        candidates = []
        for chunk in spacy_doc.noun_chunks:
            # Remove stopwords from each token of the candidate
            candidate_tokens = [token.text for token in chunk if token.text.lower() not in self.stopwords]
            candidate = " ".join(candidate_tokens).strip()
            if candidate:  # Only add non-empty candidates
                candidates.append(candidate)
        return candidates

    def encode_document(self, text):
        """
        Encode the document into input for the model with a fixed length (max_len).
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < self.max_len:
            tokens = tokens + ['[PAD]'] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        token_type_ids = [0] * self.max_len
        encoding = {
            "input_ids": torch.tensor(input_ids).unsqueeze(0).to(self.device),
            "attention_mask": torch.tensor(attention_mask).unsqueeze(0).to(self.device),
            "token_type_ids": torch.tensor(token_type_ids).unsqueeze(0).to(self.device),
        }
        return encoding

    def generate_masked_document(self, document, candidate):
        """
        Replace the candidate in the document with [MASK] tokens.
        If the candidate is not found, return None.
        """
        doc_tokens = self.tokenizer.tokenize(document)
        candidate_tokens = self.tokenizer.tokenize(candidate)
        if not candidate_tokens:
            return None
        
        mask_tokens = ['[MASK]'] * len(candidate_tokens)
        doc_text = " ".join(doc_tokens)
        candidate_text = " ".join(candidate_tokens)
        mask_text = " ".join(mask_tokens)
        
        pattern = r"\b" + re.escape(candidate_text) + r"\b"
        if not re.search(pattern, doc_text):
            return None
        
        masked_text = re.sub(pattern, mask_text, doc_text)
        masked_tokens = masked_text.split()
        if len(masked_tokens) < self.max_len:
            masked_tokens = masked_tokens + ['[PAD]'] * (self.max_len - len(masked_tokens))
        else:
            masked_tokens = masked_tokens[:self.max_len]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        attention_mask = [1 if token != '[PAD]' else 0 for token in masked_tokens]
        token_type_ids = [0] * self.max_len
        
        masked_encoding = {
            "input_ids": torch.tensor(input_ids).unsqueeze(0).to(self.device),
            "attention_mask": torch.tensor(attention_mask).unsqueeze(0).to(self.device),
            "token_type_ids": torch.tensor(token_type_ids).unsqueeze(0).to(self.device),
        }
        return masked_encoding

    def compute_similarity(self, ori_encoding, masked_encoding):
        """
        Compute cosine similarity between the embedding of the original document and the masked document.
        If the model has pooler_output, use it; otherwise, use mean pooling on last_hidden_state.
        """
        with torch.no_grad():
            ori_outputs = self.model(**ori_encoding)
            masked_outputs = self.model(**masked_encoding)
        ori_embed = (ori_outputs.pooler_output if hasattr(ori_outputs, "pooler_output") and ori_outputs.pooler_output is not None 
                     else torch.mean(ori_outputs.last_hidden_state, dim=1))
        masked_embed = (masked_outputs.pooler_output if hasattr(masked_outputs, "pooler_output") and masked_outputs.pooler_output is not None 
                        else torch.mean(masked_outputs.last_hidden_state, dim=1))
        cosine_sim = F.cosine_similarity(ori_embed, masked_embed)
        return cosine_sim.item()

    def extract_keyphrases(self, document, top_k=10):
        """
        Main function: extract keyphrases from the document.
        """
        doc = self.clean_text(document)
        ori_encoding = self.encode_document(doc)
        candidates = self.extract_candidates(doc)
        
        scored_candidates = []
        for candidate in candidates:
            masked_encoding = self.generate_masked_document(doc, candidate)
            if masked_encoding is None:
                continue
            score = self.compute_similarity(ori_encoding, masked_encoding)
            scored_candidates.append((candidate, score))
        
        scored_candidates = sorted(scored_candidates, key=lambda x: x[1])
        
        seen = set()
        keyphrases = []
        for cand, score in scored_candidates:
            if cand.lower() in seen:
                continue
            seen.add(cand.lower())
            keyphrases.append(cand)
            if len(keyphrases) >= top_k:
                break
        return keyphrases
    

class AsyncMDERank:
    """
    
    Main steps:
      1. Preprocess the document.
      2. Extract candidate keyphrases using spaCy (based on noun_chunks).
      3. For each candidate, replace (mask) the candidate in the document and encode it as input for the model.
      4. Compute cosine similarity between the embedding of the original document and the masked document.
      5. Rank the candidates by score (lower scores indicate higher importance of the candidate).
    """
    def __init__(self, model_name="avsolatorio/GIST-small-Embedding-v0", 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 max_len=512):
        self.device = device
        self.max_len = max_len

        # Load tokenizer and model from transformers (using AutoModel)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize spaCy pipeline (using English model)
        self.spacy_nlp = spacy.load("en_core_web_sm")
        
        # Load English stopwords
        self.stopwords = set(stopwords.words('english'))
        
        # If needed, we still keep NLTK's RegexpParser to handle special cases.
        self.grammar = "NP: {<NN.*|JJ>*<NN.*>}"
        self.np_parser = nltk.RegexpParser(self.grammar)

    async def clean_text(self, text):
        """
        Preprocess text: remove special characters, replace tabs, and remove extra whitespace.
        """
        def _clean():
            text2 = re.sub(r'[<>[\]{}]', ' ', text)
            text2 = text2.replace("\t", " ")
            text2 = re.sub(r'\s{2,}', ' ', text2)
            return text2.strip()
        return await asyncio.to_thread(_clean)

    async def extract_candidates(self, document):
        """
        Extract candidate keyphrases using spaCy:
          - Use spaCy to tokenize and POS-tag the text.
          - Extract noun_chunks (noun phrases) from the text.
        """
        doc_clean = await self.clean_text(document)
        def _extract(doc):
            spacy_doc = self.spacy_nlp(doc)
            candidates = []
            for chunk in spacy_doc.noun_chunks:
                # Remove stopwords from each token of the candidate
                candidate_tokens = [token.text for token in chunk if token.text.lower() not in self.stopwords]
                candidate = " ".join(candidate_tokens).strip()
                if candidate:  # Only add non-empty candidates
                    candidates.append(candidate)
            print("Candidates:", candidates)
            return candidates
        return await asyncio.to_thread(_extract, doc_clean)

    async def encode_document(self, text):
        """
        Encode the document into input for the model with a fixed length (max_len).
        """
        def _encode():
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) < self.max_len:
                tokens = tokens + ['[PAD]'] * (self.max_len - len(tokens))
            else:
                tokens = tokens[:self.max_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
            token_type_ids = [0] * self.max_len
            encoding = {
                "input_ids": torch.tensor(input_ids).unsqueeze(0).to(self.device),
                "attention_mask": torch.tensor(attention_mask).unsqueeze(0).to(self.device),
                "token_type_ids": torch.tensor(token_type_ids).unsqueeze(0).to(self.device),
            }
            return encoding
        return await asyncio.to_thread(_encode)

    async def generate_masked_document(self, document, candidate):
        """
        Replace the candidate in the document with [MASK] tokens.
        If the candidate is not found, return None.
        """
        def _generate():
            doc_tokens = self.tokenizer.tokenize(document)
            candidate_tokens = self.tokenizer.tokenize(candidate)
            if not candidate_tokens:
                return None
            
            mask_tokens = ['[MASK]'] * len(candidate_tokens)
            doc_text = " ".join(doc_tokens)
            candidate_text = " ".join(candidate_tokens)
            mask_text = " ".join(mask_tokens)
            
            pattern = r"\b" + re.escape(candidate_text) + r"\b"
            if not re.search(pattern, doc_text):
                return None
            
            masked_text = re.sub(pattern, mask_text, doc_text)
            masked_tokens = masked_text.split()
            if len(masked_tokens) < self.max_len:
                masked_tokens = masked_tokens + ['[PAD]'] * (self.max_len - len(masked_tokens))
            else:
                masked_tokens = masked_tokens[:self.max_len]
            
            input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            attention_mask = [1 if token != '[PAD]' else 0 for token in masked_tokens]
            token_type_ids = [0] * self.max_len
            
            masked_encoding = {
                "input_ids": torch.tensor(input_ids).unsqueeze(0).to(self.device),
                "attention_mask": torch.tensor(attention_mask).unsqueeze(0).to(self.device),
                "token_type_ids": torch.tensor(token_type_ids).unsqueeze(0).to(self.device),
            }
            return masked_encoding
        return await asyncio.to_thread(_generate)

    async def compute_similarity(self, ori_encoding, masked_encoding):
        """
        Compute cosine similarity between the embedding of the original document and the masked document.
        If the model has pooler_output, use it; otherwise, use mean pooling on last_hidden_state.
        """
        def _compute():
            with torch.no_grad():
                ori_outputs = self.model(**ori_encoding)
                masked_outputs = self.model(**masked_encoding)
            ori_embed = (ori_outputs.pooler_output if hasattr(ori_outputs, "pooler_output") and ori_outputs.pooler_output is not None 
                         else torch.mean(ori_outputs.last_hidden_state, dim=1))
            masked_embed = (masked_outputs.pooler_output if hasattr(masked_outputs, "pooler_output") and masked_outputs.pooler_output is not None 
                         else torch.mean(masked_outputs.last_hidden_state, dim=1))
            cosine_sim = F.cosine_similarity(ori_embed, masked_embed)
            return cosine_sim.item()
        return await asyncio.to_thread(_compute)

    async def extract_keyphrases(self, document, top_k=10):
        """
        Main function: extract keyphrases from the document.
        """
        doc = await self.clean_text(document)
        ori_encoding = await self.encode_document(doc)
        candidates = await self.extract_candidates(doc)
        
        async def process_candidate(candidate):
            masked_encoding = await self.generate_masked_document(doc, candidate)
            if masked_encoding is None:
                return None
            score = await self.compute_similarity(ori_encoding, masked_encoding)
            return candidate, score
        
        tasks = [process_candidate(candidate) for candidate in candidates]
        results = await asyncio.gather(*tasks)
        scored_candidates = [res for res in results if res is not None]
        scored_candidates = sorted(scored_candidates, key=lambda x: x[1])
        
        seen = set()
        keyphrases = []
        for cand, score in scored_candidates:
            if cand.lower() in seen:
                continue
            seen.add(cand.lower())
            keyphrases.append(cand)
            if len(keyphrases) >= top_k:
                break
        return keyphrases

# Example usage:
if __name__ == "__main__":
    # An example document
    document = ("""Samson Mow Says Bitcoin Bear Trap: What’s Next at $82,516.97? We believe in full transparency with our readers . Some of our content includes affiliate links , and we may earn a commission through these partnerships . However , this potential compensation never influences our analysis , opinions , or reviews . Our editorial content is created independently of our marketing partnerships , and our ratings are based solely on our established evaluation criteria . Read More Bitcoin has slipped to $82,516.97 , down 1.55% in 24 hours , triggering concerns of a deeper correction . But according to Samson Mow  , this breakdown is a “bear trap”—a fakeout designed to flush out weak hands before a larger move higher. Mow remains firm on his $1 million BTC target , arguing the recent sell-off doesn’t reflect fundamentals . Bitcoin still holds a $1.64 trillion market cap , with 19.84 million BTC in circulation. While Mow sees upside ahead , charts tell a different story . BTC has broken below a symmetrical triangle , with the former support at $83,650 now acting as resistance. Bitcoin Technical Setup Signals Caution The recent breakdown from the symmetrical triangle pattern has turned $83,650 into a key resistance zone , stalling any immediate rebound . A bearish engulfing candle under this level signals continued selling pressure. Current Price : $82,516.97 - 24H Volume : $19.93B - Resistance Levels : $83,650 , $85,231 , $86,841 - Support Levels : $82,000 , $81,278 , $79,990 - 50 EMA : $85,231 (above current price) - RSI (14) : 27.63 (oversold) - The RSI remains oversold , but without bullish divergence , offering no clear sign of reversal . A breakdown through the triple bottom around $83,000 further weakens the structure , placing $81,278 and $79,990 in view . Volume near current levels is also subdued , reflecting a lack of strong buyer support. What Comes Next for Bitcoin? The key question is whether this is the trap Mow describes , or the start of a broader correction . Broader sentiment is mixed , with macroeconomic pressure and tight liquidity weighing on high-risk assets. A confirmed reclaim of $83,650 , followed by a breakout above the 50 EMA at $85,231 , would be an early signal of bullish recovery . Otherwise , continued failure at current levels risks a slide below $80,000. Key Signals to Monitor: Break and close above $83,650 - RSI divergence or recovery above 30 - Trading volume increase on bounce attempts - Support holding at $81,278 or $79,990 - Until these conditions are met , Bitcoin remains vulnerable . Whether Samson Mow’s call plays out will depend on how markets respond in the days ahead. BTC Bull: Earn Bitcoin Rewards with the Hottest Crypto Presale BTC Bull ($BTCBULL) is making waves as a community-driven token that automatically rewards holders with real Bitcoin when BTC hits key price milestones . Unlike traditional meme tokens , BTCBULL is built for long-term investors , offering real incentives through airdropped BTC rewards and staking opportunities. Staking & Passive Income Opportunities BTC Bull offers a high-yield staking program with an impressive 119% APY , allowing users to generate passive income . The staking pool has already attracted 882.5 million BTCBULL tokens , highlighting strong community participation. Latest Presale Updates: Current Presale Price: $0.002425 per BTCBULL - Total Raised: $4M / $4.5M target - With demand surging , this presale provides an opportunity to acquire BTCBULL at early-stage pricing before the next price increase.""")
    
    # Initialize MDERank with GPU (if available)
    mde_rank = MDERank()
    keyphrases = mde_rank.extract_keyphrases(document, top_k=50)
    print("Extracted keyphrases:", keyphrases)
    
    # Close StanfordCoreNLP connection after use
    mde_rank.nlp.close()
