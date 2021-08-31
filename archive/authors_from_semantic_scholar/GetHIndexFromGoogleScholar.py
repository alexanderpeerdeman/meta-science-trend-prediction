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

# ## Retrieving h-index from Google Scholar (using `scholarly`)

# +
#from scholarly import scholarly

# +
#author_list = list(scholarly.search_author("Stephen Hawking"))
#len(author_list)

# +
#author_list

# +
#scholarly_meta = scholarly.fill(author_list[0], sections=["indices"])
#scholarly_meta

# +
#scholarly_meta["hindex"]
# -

# ## Retrieving h-index from semantic scholar author page (rudimentary html parser)
# _dirty hack_

from bs4 import BeautifulSoup
import requests
import os
import time
import json


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
                author["publications"] = int(elem.text)

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

16694

fails_parser = []
fails_id = []

for idx, jf in enumerate(fails_parser):
    author_id = os.path.splitext(jf)[0]
    print("{:15d} Processing {} ... ".format(idx, author_id), end="")
    try:
        author_metrics = call_api(author_id)
        print("OK", end="")
    except ValueError:
        print("FAIL (parser)")
        fails_parser.append(author_id)
        continue
    except NameError:
        print("FAIL (ID)")
        fails_id.append(author_id)
        continue
        
    
    with open(author_jsons_path + jf + ".json") as filebuffer:        
        author = json.load(filebuffer)
    if "h-index" not in author.keys():
        author.update(author_metrics)
        with open(author_jsons_path + jf + ".json", "w") as filebuffer:
            json.dump(author, filebuffer)
        print(", UPDATED", end="")
    print("")

len(fails_parser)

len(fails_id)

print(fails_parser)

print(fails_id)

fails_parser_old = fails_parser

fails_id_old = fails_id


