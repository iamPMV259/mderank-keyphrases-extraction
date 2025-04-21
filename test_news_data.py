from hmac import new
from tracemalloc import start
from requests.models import Response


import requests
import time
import json

def extract_keyphrases_mderank(text: str, top_k: int) -> list[str]:
    """
    Extract keyphrases from the given text using MDERank.
    
    :param text: The input text from which to extract keyphrases.
    :param top_k: The number of top keyphrases to return.
    :return: A list of extracted keyphrases.
    """
    res: Response = requests.post("http://localhost:5040/mderank", headers={"accept": "application/json", "Content-Type": "application/json"}, json={
        "text": text,
        "top_k": top_k
    })

    return res.json().get('keyphrases', [])


if __name__ == "__main__":

    file_path = "/home/pmv259/Documents/a-star/mde-rank/data/rss_data_cleaned.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cnt = 0

    for record in data:
        cnt += 1
        title = record.get('title')
        _id = record.get('_id')
        content = record.get('content')
        multipartite_keywords = record.get('multipartite_keywords')
        bert_keywords = record.get('bert_keywords') 

        start_time: float = time.time()
        doc = title + ". " + content
        mde_keywords = extract_keyphrases_mderank(doc, top_k=15)

        print("Order: ", cnt)
        # print("ID: ", _id)
        print("Title: ", title)
        print("Content: ", content)
        print("-----------------------------------")
        print("Multipartite Keywords: ", multipartite_keywords)
        print("------------------------------------")
        print("BERT Keywords: ", bert_keywords)
        print("------------------------------------")
        print("MDERank Keywords: ", mde_keywords)
        print("MDERank Running time = ", time.time() - start_time)
        # print("===================================")
        print()
        print()
        # new_records = {
        #     "_id": _id,
        #     "title": title,
        #     "content": content,
        #     "multipartite_keywords": multipartite_keywords,
        #     "bert_keywords": bert_keywords,
        #     "mde_keywords": mde_keywords,
        #     "mde_running_time": time.time() - start_time
        # }
        if cnt == 100:
            break