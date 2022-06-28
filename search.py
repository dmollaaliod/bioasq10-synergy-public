"""Search and save the results"""

import requests
import json
import progressbar

def search(inputfile="BioASQ-taskSynergy_v2022-testset1.json",
           outputfile="testset1_search_results.json",
           num_hits=200):

    url = "http://bioasq.org:8008/cord"
    session_url = requests.get(url).text

    with open(inputfile) as f:
        questions = json.load(f)['questions']

    result = {}

    bar = progressbar.ProgressBar(max_value=len(questions))

    for i, q in enumerate(questions):
        query = q['body']
        print()
        print(query)
        api_result = requests.get(session_url, data={'json':json.dumps({'findArticles': [query, 0, num_hits]})})
        result_json = json.loads(api_result.text)

        if result_json and 'result' in result_json:
            result[q['id']] = result_json['result']

        with open(outputfile, 'w') as f:
            json.dump(result, f, indent=2)

        
        bar.update(i)


if __name__ == "__main__":
    search()
