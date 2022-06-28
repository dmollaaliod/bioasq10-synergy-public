"""Run  MQ-1 for the Synergy task"""

import os
import json

from multisummarise import qsummarise as single_summarise
from multisummarise import bioasq_summarise as multi_summarise
from nnc import DistilBERT

import search

TESTSETFILE = "BioASQ-taskSynergy_v2022-testset4.json"
SEARCHRESULTSFILE = "testset4_search_results.json"
FEEDBACKFILE = "BioASQ-taskSynergy_2022-feedback_round4.json"
OUTPUTFILE = 'round4_mq1.json'

nnc = DistilBERT(hidden_layer_size=50, positions=True)
nnc.load("task10b_distilbert_model_32.pt")
#nnc.fit(None, None, None, restore_model=True, verbose=0, savepath="task9b_albert_model_32.pt")

# Retrieve the documents
if not os.path.exists(SEARCHRESULTSFILE):
    print("Retrieving the documents")
    search.search(inputfile=TESTSETFILE,
                  outputfile=SEARCHRESULTSFILE,
                  num_hits=200)
with open(SEARCHRESULTSFILE) as f:
    search_results = json.load(f)

nanswers={"summary": 6,
          "factoid": 2,
          "yesno": 2,
          "list": 3}

with open(TESTSETFILE) as f:
    questions = json.load(f)['questions']

with open(FEEDBACKFILE) as f:
    feedback = json.load(f)['questions']

print("Processing %i questions" % len(questions))
json_results = []
for i, q in enumerate(questions):
    print("Question", i, "of", len(questions))
    feedback_found = False
    question_feedback = None
    gold_document_negatives = []
    gold_snippet_negatives = []
    #print(q['body'])
    #print('Answer required:', q['answerReady'])
    for f in feedback:
        if q['id'] == f['id']:
            print("Feedback found")
            feedback_found = True
            question_feedback = f
            gold_document_negatives = [d['id'] for d in question_feedback['documents'] if not d['golden']]
            gold_snippet_negatives = [s['text'] for s in question_feedback['snippets'] if not s['golden']]
            gold_snippet_positives = [s['text'] for s in question_feedback['snippets'] if s['golden']]

            #print("Gold snippet negatives:", gold_snippet_negatives)

    json_r = {'body': q['body'],
              'id': q['id'],
              'type': q['type'],
              'answerReady': q['answerReady']}
    question = q['body']
    print(question)

    if q['id'] in search_results:
        result = search_results[q['id']]
        
        print("Processing %i results from a total of %i" % (result['articlesPerPage'], result['size']))
        print(len(result['documents']))

        json_r['documents'] = [d['cord_uid'] for d in result['documents']]
        snippets = []
        for d in result['documents']:
            if d['cord_uid'] in gold_document_negatives:
                # print("Ignoring document negative", d['cord_uid'])
                continue

            single_summary = single_summarise(question, d['documentAbstract'])
            for s in single_summary:
                result_summary = {
                    'document': d['cord_uid'],
                    'beginSection': 'abstract',
                    'endSection': 'abstract',
                    'offsetInBeginSection': s[0][0],
                    'offsetInEndSection': s[0][1],
                    'text': d['documentAbstract'][s[0][0]:s[0][1]]
                }
                snippets.append(result_summary)
        json_r['snippets'] = snippets

#        if q['answerReady']:
        if True:
#            input_sentences = gold_snippet_positives + [s['text'] for s in snippets if s['text'] not in gold_snippet_positives+gold_snippet_negatives]
            input_sentences = [s['text'] for s in snippets if s['text'] not in gold_snippet_negatives]
            m_summary = multi_summarise(question, input_sentences, nnc=nnc, n=nanswers[q['type']])
            print(m_summary)
            m_summary_text = ' '.join([t for t, n, score in m_summary])
            json_r['ideal_answer'] = m_summary_text
            if q['type'] == 'yesno':
                json_r['exact_answer'] = 'yes'
            else:
                json_r['exact_answer'] = []

        # Remove gold data and truncate lists to conform with requirements
        if feedback_found:
            print("from %i, documents", len(json_r['documents']))
            documents_gold = [d['id'] for d in question_feedback['documents']]
            json_r['documents'] = [d for d in json_r['documents'] if d not in documents_gold]
            print("to %i, documents", len(json_r['documents']))
            print("from %i, snippets", len(json_r['snippets']))
            snippets_gold = [d['text'] for d in question_feedback['snippets']]
            json_r['snippets'] = [d for d in json_r['snippets'] if d['text'] not in snippets_gold]
            print("to %i, snippets", len(json_r['snippets']))
        json_r['documents'] = json_r['documents'][:10]
        json_r['snippets'] = json_r['snippets'][:10]
    else:
        print("No results found; ignoring the question")
        json_r['documents'] = []
        json_r['snippets'] = []
        if q['type'] == 'yesno':
            json_r['exact_answer'] = 'yes'
        else:
            json_r['exact_answer'] = []
        json_r['ideal_answer'] = ''

    json_results.append(json_r)

result = {'questions': json_results}
with open(OUTPUTFILE,'w') as f:
    json.dump(result, f, indent=2)
