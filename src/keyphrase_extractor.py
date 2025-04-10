from .mderank import MDERank
from .mderank import AsyncMDERank
import asyncio


async def extract_keyphrases(text, top_k=20):
    """
    Wrapper function to extract keyphrases from text.
    Returns a list of the top_k keyphrases with the lowest scores (i.e., the most important ones).
    """
    mde = AsyncMDERank()
    ranked = await mde.extract_keyphrases(text, top_k=top_k)
    return ranked
