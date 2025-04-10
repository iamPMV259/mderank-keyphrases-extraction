from datetime import datetime
import sys
import os
import asyncio
import json
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.keyphrase_extractor import extract_keyphrases
from .call_gemini_api import extract_keyphrases as gemini_extract_keyphrases




async def main():
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "rss_data_cleaned.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cnt = 0
    default_date = "2025-03-29"
    for record in data:
        


        
        print(record)
        title = record.get('title')
        id = record.get('_id')
        published_time = record.get('published_time')
        content = record.get('content')
        multipartite_keywords = record.get('multipartite_keywords')
        bert_keywords = record.get('bert_keywords')        

        if not published_time.startswith("2025-03-29"):
            continue
        
        cnt += 1
        
        print(cnt)

        if cnt == 50:
            break


        # doc = title + ". " + content
        # mde_keywords = await extract_keyphrases(doc)

        # new_record = {
        #     "_id": id,
        #     "title": title,
        #     "published_time": published_time,
        #     "content": content,
        #     "multipartite_keywords": multipartite_keywords,
        #     "bert_keywords": bert_keywords
        # }
        
        # records_path = os.path.join(os.path.dirname(__file__), "..", "records")
        # with open(os.path.join(records_path, f"{id}.json"), "w", encoding="utf-8") as f:
        #     json.dump(new_record, f, ensure_ascii=False, indent=4)

        # print(f"ID: {id}")
        # print(f"Content: {content}")
        # print("----------------------------")
        # print(f"Multipartite Keywords: {multipartite_keywords}")
        # print("----------------------------")
        # print(f"BERT Keywords: {bert_keywords}")
        print("----------------------------")
        # print(f"MDE Keywords: {mde_keywords}")
        print("----------------------------")
        print()
        print()
    


if __name__ == "__main__":
    asyncio.run(main())
