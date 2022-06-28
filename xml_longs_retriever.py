from xml.dom.minidom import parse
import os

from xml_abstract_retriever import getText, getAbstract

TEST_QUESTIONS_PATH = os.path.join("..","devtestset-questions.txt")
ABSTRACTS_PATH = os.path.join("..")
CORPUS_PATH = os.path.join("..","ClinicalInquiries.xml")

def getLongs(q_path=TEST_QUESTIONS_PATH, 
             a_path=ABSTRACTS_PATH, 
             corpus_path=CORPUS_PATH, verbose=1):
    """Yield the question, abstract, and target summaries
    >>> longs = [l for l in getLongs(verbose=0)]
    Processing 78 questions
    >>> len(longs)
    658
    >>> print longs[0][0], longs[0][1][:20], longs[0][2][:20], longs[2][3][:20]
    6495 Is there much risk i A large multicenter, The use of quinolone
"""
    # Build DOM dictionary of questions
    d = parse(corpus_path)
    q_dom = dict()
    for qd in d.getElementsByTagName('record'):
        q_dom[qd.getAttribute('id')] = qd
        
    # Obtain questions IDs
    with open(q_path) as f:
        question_ids = [l.split()[0] for l in f.readlines()]
    print("Processing " + len(question_ids) + " questions")
    for q in question_ids:
        qd = q_dom[q]
        qtext = getText(qd.getElementsByTagName('question')[0].childNodes)
        for lng in qd.getElementsByTagName('long'):
            longtext = getText(lng.getElementsByTagName('longtext')[0].childNodes)
            for r in lng.getElementsByTagName('ref'):
                filename = os.path.join(a_path,r.getAttribute('abstract'))
                yield (q,qtext,longtext,getAbstract(filename,verbose=verbose)[0])

if __name__ == "__main__":
    import doctest
    doctest.testmod()
