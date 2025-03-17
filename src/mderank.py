import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk import pos_tag, word_tokenize

# Tải dữ liệu cần thiết cho NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class MDERank:
    def __init__(self, model_name="bert-base-uncased", pooling="max"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def compute_embedding(self, text):
        # Chuẩn hóa đầu vào và lấy output từ mô hình BERT
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Lấy hidden_states: [batch_size, sequence_length, hidden_size]
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
        Sử dụng NLTK để tách từ, gán nhãn POS và trích xuất các cụm từ ứng viên
        theo pattern: liên tục các từ có tag bắt đầu bằng JJ (tính từ) hoặc NN (danh từ).
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
        # Loại bỏ các cụm từ trùng lặp và có độ dài ít nhất 1 từ
        candidates = list(set([c for c in candidates if len(c.split()) >= 1]))
        return candidates

    def mask_text(self, text, candidate):
        """
        Thay thế các xuất hiện của candidate trong text bằng [MASK] với số lượng token tương ứng.
        """
        candidate_tokens = candidate.split()
        mask_token = " ".join(["[MASK]"] * len(candidate_tokens))
        # Sử dụng regex để thay thế, không phân biệt hoa thường
        pattern = re.compile(re.escape(candidate), re.IGNORECASE)
        masked_text = pattern.sub(mask_token, text)
        return masked_text

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

    def rank_keyphrases(self, text):
        """
        Tính toán embedding của văn bản gốc và đối với mỗi ứng viên, tính embedding của văn bản đã mask.
        Sau đó, tính cosine similarity giữa hai embedding này. Ứng viên có similarity thấp hơn (nghĩa là mất thông tin lớn)
        được xem là quan trọng hơn.
        """
        original_embedding = self.compute_embedding(text)
        candidates = self.extract_candidates(text)
        scores = {}
        for candidate in candidates:
            masked_text = self.mask_text(text, candidate)
            masked_embedding = self.compute_embedding(masked_text)
            sim = self.cosine_similarity(original_embedding, masked_embedding)
            scores[candidate] = sim
        # Sắp xếp các ứng viên theo thứ tự tăng dần của similarity
        ranked = sorted(scores.items(), key=lambda x: x[1])
        return ranked
