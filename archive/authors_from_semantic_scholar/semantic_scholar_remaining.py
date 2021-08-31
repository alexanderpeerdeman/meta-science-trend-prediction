# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from bs4 import BeautifulSoup
import requests
import os
import time
import json
from tqdm import tqdm


def call_api(author_id):    
    response = requests.get("https://www.semanticscholar.org/author/" + str(author_id))

    if response.ok:
        soup = BeautifulSoup(response.text)
        author_natural_name = soup.title.text[:-19]
        author = dict()
        
        for div in soup.find_all("div", class_="author-detail-card__stats-row"):
            label = div.find("span", class_="author-detail-card__stats-row__label")

            if label.text == "Publications":        
                elem = label
                for _ in range(2):
                    elem = elem.next
                author["publications"] = int(elem.text.replace(",", ""))

            elif label.text == "h-index":        
                elem = label
                for _ in range(5):
                    elem = elem.next
                author["h-index"] = int(elem.text)

            elif label.text == "Citations":        
                elem = label
                for _ in range(2):
                    elem = elem.next
                author["citations"] = int(elem.text.replace(",", ""))
        for key, value in author.items():
            if value < 0:
                raise ValueError("{} has value of {}.".format(key, value))

        return author
    else:
        raise NameError("[{}] Could not get respone for authorID {}.".format(response.status_code, author_id))



author_jsons_path = "data/semantic_scholar/authors/"
author_jsons = os.listdir(author_jsons_path)

not_done = []
for idx, jf in enumerate(author_jsons):
    with open(author_jsons_path + jf) as filebuffer:        
        author = json.load(filebuffer)
        
    if "h-index" not in author.keys() or "publications" not in author.keys() or "citations" not in author.keys():
        print(jf)
        not_done.append(jf)


len(not_done)

for jf in tqdm(not_done):
    with open(author_jsons_path + jf) as filebuffer:
        author = json.load(filebuffer)
        
    author_id = os.path.splitext(jf)[0]
    try:
        author_meta = call_api(author_id)
    except Exception as e:
        tqdm.write("{}".format(e))
        continue
        
    author.update(author_meta)
        
    with open(author_jsons_path + jf, "w") as filebuffer:
        json.dump(author, filebuffer)
    

with open(author_jsons_path + "1491355593.json") as f:
    print(json.load(f))
