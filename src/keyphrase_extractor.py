from .mderank import MDERank

def extract_keyphrases(text, top_k=10):
    """
    Hàm bọc để trích xuất keyphrase từ văn bản.
    Trả về danh sách top_k keyphrase có score thấp nhất (nghĩa là quan trọng nhất).
    """
    mde = MDERank()
    ranked = mde.rank_keyphrases(text)
    top_candidates = [phrase for phrase, score in ranked[:top_k]]
    return top_candidates
