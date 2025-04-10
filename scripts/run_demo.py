import sys
import os
import asyncio
import nltk
from nltk.stem import WordNetLemmatizer
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

nltk.download("omw-1.4")

from src.keyphrase_extractor import extract_keyphrases

def lemma(keyword: str, word_lem: WordNetLemmatizer = WordNetLemmatizer()):
    """Helper function to lemmatize keyword, except several tokens"""
    lst = keyword.split()
    new_txt = ""
    outliers = [
        "dogs",
        "us",
        "mas",
    ]  # HACK: there are some outliers that make the keyword incomprehensible: `us` turns to `u` after lemmatization
    for i in lst:
        if i not in outliers:
            j = " " + word_lem.lemmatize(i, pos="n")
            
            new_txt += j
        else:
            new_txt += i
    return new_txt[1:]


async def main():
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    keyphrases = await extract_keyphrases(text)
    # print("Extracted Keyphrases:")
    # for i, kp in enumerate(keyphrases, start=1):
    #     print(f"{i}. {kp}")
    keyphrases = [lemma(kw) for kw in keyphrases]
    print("Extracted Keyphrases:")
    for i, kp in enumerate(keyphrases, start=1):
        print(f"{i}. {kp}")



if __name__ == "__main__":
    asyncio.run(main())
