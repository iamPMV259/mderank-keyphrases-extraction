import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.keyphrase_extractor import extract_keyphrases

async def main():
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    keyphrases = await extract_keyphrases(text)
    print("Extracted Keyphrases:")
    for i, kp in enumerate(keyphrases, start=1):
        print(f"{i}. {kp}")

if __name__ == "__main__":
    asyncio.run(main())
