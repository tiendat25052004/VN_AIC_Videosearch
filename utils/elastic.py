
import requests
from requests.auth import HTTPBasicAuth
import json
import numpy as np
URL_SEARCH = {
    "audio": "http://elastic:changeme@20.2.84.151:9200/audio/_search",
    "ocr": "http://elastic:changeme@20.2.84.151:9200/ocr/_search"
}
headers = {
  'Content-Type': 'application/json'
}

def extract_ans(ans: dict):
    hits = ans['hits']['hits']
    results = list(map(lambda x: x['_source'], hits))
    lst_scores = [hit["_score"] for hit in hits]
    
    for i, result in enumerate(results):
        result["score"] = lst_scores[i]
    return results
    

def normal_search(s: str, index: str = 'ocr'):
    """
        dùng cho các câu đầy đủ, đơn giản, không dấu cũng được, hoa thường không quan trọng
        
    """
    
    payload = {
        "query": {
            "match": {
                f"{index}" : {
                    "query": s
                }
            }
        }
    }
    
    response = requests.request("GET", URL_SEARCH[index], headers=headers, data=json.dumps(payload), auth=HTTPBasicAuth('elastic', 'sang'))
    return extract_ans(response.json())

def fuzzy_normal_search(s: str, fuzzy: str = 'AUTO', index: str = 'ocr'):
    """
        dùng cho các câu đầy đủ, đơn giản, không dấu cũng được, hoa thường không quan trọng
        co the viet sai vai tu, fuzzy là 0 1 hoặc 2, thử xem cái nào ổn, 
        mặc định là auto
        chỉ nên search one word hoặc ít sai
        
    """
    
    payload = {
        "query": {
            "fuzzy": {
                f"{index}": {
                        "fuzziness": fuzzy,
                        "value": s
                                    }
                        }
                    }
            }   
    
    
    response = requests.request("GET", URL_SEARCH[index], headers=headers, data=json.dumps(payload), auth=HTTPBasicAuth('elastic', 'sang'))
    return extract_ans(response.json())


def advance_query(s, slop: int=3, inorder = True, fuzzyness: str = 'AUTO', index: str = 'ocr'):
    print(f"Searching for: {s}")
    """
        Tìm kiếm nâng cao, tách từng từ ra sau đó search và ghép lại, slop là khoảng cách tối đa để ghép, càng để rộng thì
        càng tìm được nhiều nhưng có thể bị tối nghĩa
        
        ví dụ nếu bạn tìm con chó và để slop quá lớn nó có thể tìm ra "con người không phải là chó" ở điểm cao hơn,
        mà thật ra nó nó có chữ chó :)) nhưng không hợp nghĩa con chó lắm
        để khoangg 2 là đẹp
        inorder là các từ được tìm phải đúng thứ tự, nếu tìm dài thì nên bật
        mà thử tắt xem coi ra đúng hơn không
        fuzzyness thì như bên trên, 0 1 2 hoặc auto
    """
    
    val = list(map(lambda x: x.strip(), s.split(' ')))
    payload = {
     "query": {
            "span_near" : {
                "clauses" : [
                   { 
                   	"span_multi": {
                           "match": {
                                 "fuzzy": {
                                    f"{index}": {
                                           "fuzziness": fuzzyness,
                                            "value": val[i]
                                                      }
                                            }
                                      }
                               }
                   }
                    for i in range(len(val))
                   
                ],
                "slop" : slop,
                "in_order" : inorder
            }
        },
      
        }
    
    
    response = requests.request("GET", URL_SEARCH[index], headers=headers, data=json.dumps(payload), auth=HTTPBasicAuth('elastic', 'sang'))
    print(extract_ans(response.json()))
    return extract_ans(response.json())

# print(advance_query("gim chua thit hai", fuzzyness='2', inorder=False, slop=2))