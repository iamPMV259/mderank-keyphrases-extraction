from .mderank import AsyncMDERank
import asyncio

async def extract_keyphrases(text, top_k=50):
    """
    Wrapper function to extract keyphrases from text.
    Returns a list of the top_k keyphrases with the lowest scores (i.e., the most important ones).
    """
    mde = AsyncMDERank()
    text = text.lower()
    ranked = await mde.rank_keyphrases(text)
    top_candidates = [phrase for phrase, score in ranked[:top_k]]
    return top_candidates
