#! /usr/bin/python
"""Summarise API:
Private
Version 0.3, 06-OCT-2014

This is a summarising module which summarises the text/document
returned as the result of a query. It returns the n best sentences
which best summarise the text. It makes use of the sklearn, numpy and
nltk modules.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from nltk import sent_tokenize
#from nltk.corpus import stopwords
import os
import sys
import codecs
#import pickle
import scipy.sparse

sys.path.append('/code/dev/')
sys.path.append('../../dev/')
sys.path.append(os.path.dirname(__file__))

from xml_longs_retriever import getLongs
from my_tokenizer import my_tokenize
#import word2vec

#stopwords_english = stopwords.words('english')
stopwords_english = []

TEST_QUESTIONS_PATH = "devtestset-questions.txt"
ABSTRACTS_PATH = os.path.join("..","..","dev")
CORPUS_PATH = os.path.join("..","..","dev","ClinicalInquiries.xml")

def evaluateSummaries(summariser,
                      q_path=TEST_QUESTIONS_PATH, a_path=ABSTRACTS_PATH,
                      xml_rouge_filename = 'rouge.xml'):
    """Evaluate the summariser"""
    f = open(xml_rouge_filename,'w')
    f.write('<ROUGE-EVAL version="1.0">\n')
    rouge_i = 0
    for (qid,question,target,abstract) in getLongs(q_path=TEST_QUESTIONS_PATH,
                                                   a_path=ABSTRACTS_PATH,
                                                   corpus_path=CORPUS_PATH,
                                                   verbose=0):
        rouge_i += 1

        # XML data
        f.write("""<EVAL ID="%i">
 <MODEL-ROOT>
 rouge/models
 </MODEL-ROOT>
 <PEER-ROOT>
 rouge/peers
 </PEER-ROOT>
 <INPUT-FORMAT TYPE="SPL" />
 <MODELS>
   <M ID="1">model%i</M>
 </MODELS>
 <PEERS>
   <P ID="A">peer%i</P>
 </PEERS>
</EVAL>
""" % (rouge_i,rouge_i,rouge_i))

        # Model
        with codecs.open('rouge/models/model%i' % rouge_i, 'w', 'utf-8') as fout:
            fout.write('\n'.join(sent_tokenize(target))+'\n')

        # Peer
        with codecs.open('rouge/peers/peer%i' % rouge_i,'w', 'utf-8') as fout:
            if abstract == '':
                fout.write('\n')
            else:
                for ((b,e),score) in summariser(question,abstract):
                    fout.write(abstract[b:e]+'\n')

    f.write('</ROUGE-EVAL>\n')
    f.close()
    ROUGE_CMD = 'perl rouge/ROUGE-1.5.5.pl -e rouge/data -a -n 2 -2 4 -U %s' % (xml_rouge_filename)
    print("Calling " + ROUGE_CMD)
    os.system(ROUGE_CMD)

def summarise(question, text, n=3):
    return qsummarise(question, text, n)

def rsummarise(question,text,n=3):
    """Regression-based summarisation

    >>> question = "What is the best treatment for migraines?"
    >>> text = "Exome results are reported for two patients with connective tissue dysplasia, one refining a clinical diagnosis of Ehlers-Danlos to Marfan syndrome, the other suggesting arthrogryposis derived from maternofetal Stickler syndrome. Patient 1 had mutations in transthyretin (TTR), fibrillin (FBN1), and a calcium channel (CACNA1A) gene suggesting diagnoses of transthyretin amyloidosis, Marfan syndrome, and familial hemiplegic migraines, respectively. Patient 2 presented with arthrogryposis that was correlated with his mother's habitus and arthritis once COL2A1 mutations suggestive of Stickler syndrome were defined. Although DNA results often defy prediction by the best of clinicians, these patients illustrate needs for ongoing clinical scholarship (e.g., to delineate guidelines for management of mutations like that for hyperekplexia in Patient 2) and for interpretation of polygenic change that is optimized by clinical genetic/syndromology experience (e.g., suggesting acetazolamide therapy for Patient 1 and explaining arthrogryposis in Patient 2)."
    >>> for s in rsummarise(question,text,n=3):
    ...   print s[0], round(s[1],3)
    (230, 449) 0.148
    (450, 617) 0.114
    (618, 1057) 0.129
"""
    global tfidf, regression
    featuresquestion = tfidf.transform([question])
    sentences = sent_tokenize(text)
    featuressentences = tfidf.transform(sentences)
    distances = pairwise_distances(featuresquestion,
                                   featuressentences,'cosine').transpose()
    allfeatures = scipy.sparse.hstack((distances,featuressentences))
    predictions = regression.predict(allfeatures)
    scores = zip(predictions,range(len(predictions)))
    scores.sort()
    summary = scores[-n:]

    # Find the character offsets
    offsets = []
    begin = 0
    for s in sentences:
        b = text.find(s,begin)
        assert(b >= 0)
        offsets.append((b,b+len(s)))
        begin = offsets[-1][1]

    # Return the results
    summary.sort(cmp = lambda x,y: cmp(x[1],y[1]))
    return [(offsets[i],score) for (score,i) in summary]


def vsummarise(question,text,n=3):
    """simple query-based summarisation.

    Performs cosine similarity between word2vec of the question and word2vec of the candidate sentences.

    >>> question = "What is the best treatment for migraines?"
    >>> text = "Exome results are reported for two patients with connective tissue dysplasia, one refining a clinical diagnosis of Ehlers-Danlos to Marfan syndrome, the other suggesting arthrogryposis derived from maternofetal Stickler syndrome. Patient 1 had mutations in transthyretin (TTR), fibrillin (FBN1), and a calcium channel (CACNA1A) gene suggesting diagnoses of transthyretin amyloidosis, Marfan syndrome, and familial hemiplegic migraines, respectively. Patient 2 presented with arthrogryposis that was correlated with his mother's habitus and arthritis once COL2A1 mutations suggestive of Stickler syndrome were defined. Although DNA results often defy prediction by the best of clinicians, these patients illustrate needs for ongoing clinical scholarship (e.g., to delineate guidelines for management of mutations like that for hyperekplexia in Patient 2) and for interpretation of polygenic change that is optimized by clinical genetic/syndromology experience (e.g., suggesting acetazolamide therapy for Patient 1 and explaining arthrogryposis in Patient 2)."
    >>> for s in vsummarise(question,text,n=3):
    ...   print(s[0] + "," + round(s[1],4))
    (0, 229) 0.7209
    (450, 617) 0.7089
    (618, 1057) 0.4658
"""
    # Tfidf to weight the features
    tfidf = TfidfVectorizer(tokenizer=my_tokenize)
    sentences = sent_tokenize(text)
    tfidf.fit(sentences+[question])
    tfidf_question = tfidf.transform([question]).todense()
    tokens_question = my_tokenize(question)
    vocabulary = tfidf.get_feature_names()

    tfidf_sentences = tfidf.transform(sentences).todense()

    # Obtain the question features
    vectors_question = word2vec.vectors(tokens_question)
    clean_vquestion = []
    question_weights = []
    for i in range(len(tokens_question)):
        if vectors_question[i] != None:
            clean_vquestion.append(vectors_question[i])
#            question_weights.append(tfidf_question[0,vocabulary.index(tokens_question[i])])
            question_weights.append(1)

    word2vec_question = np.average(clean_vquestion,
                                   axis=0,
                                   weights=question_weights)

    # Score each text sentence
    sentences = sent_tokenize(text)
    distances = []
    for i in range(len(sentences)):
        s = sentences[i]
        tokens_sentence = my_tokenize(s)
        w2v = word2vec.vectors(tokens_sentence)
        clean_w2v = []
        sentence_weights = []
        for j in range(len(tokens_sentence)):
            if w2v[j] != None:
                clean_w2v.append(w2v[j])
#                sentence_weights.append(tfidf_sentences[i,vocabulary.index(tokens_sentence[j])])
                sentence_weights.append(1)

        if len(sentence_weights) == 0:
            # 0 score for sentences without word vectors
            distances.append(np.inf)
        else:
            vec = np.average(clean_w2v,axis=0,weights=sentence_weights)
            distances.append(pairwise_distances([word2vec_question],
                                                [vec],
                                                'cosine'))
    scores = [(distances[i],i) for i in range(len(sentences))]

    # Obtain the top n sentences
    scores.sort()
    summary = scores[:n]

    # Find the character offsets
    offsets = []
    begin = 0
    for s in sentences:
        b = text.find(s,begin)
        assert(b >= 0)
        offsets.append((b,b+len(s)))
        begin = offsets[-1][1]

    # Return the results
    summary.sort(cmp = lambda x,y: cmp(x[1],y[1]))
    return [(offsets[i],score) for (score,i) in summary]


# with open('pickles/trainset_sents_tfidf.pickle') as f:
#     tfidf = cPickle.load(f)
# with open('pickles/trainset_sents_lsa200.pickle') as f:
#     pca = cPickle.load(f)

def qsummarise(question,text,n=3):
    """simple query-based summarisation.

    Performs cosine similarity between tf.idf of the question and tf.idf of the candidate sentences.

    >>> question = "What is the best treatment for migraines?"
    >>> text = "Exome results are reported for two patients with connective tissue dysplasia, one refining a clinical diagnosis of Ehlers-Danlos to Marfan syndrome, the other suggesting arthrogryposis derived from maternofetal Stickler syndrome. Patient 1 had mutations in transthyretin (TTR), fibrillin (FBN1), and a calcium channel (CACNA1A) gene suggesting diagnoses of transthyretin amyloidosis, Marfan syndrome, and familial hemiplegic migraines, respectively. Patient 2 presented with arthrogryposis that was correlated with his mother's habitus and arthritis once COL2A1 mutations suggestive of Stickler syndrome were defined. Although DNA results often defy prediction by the best of clinicians, these patients illustrate needs for ongoing clinical scholarship (e.g., to delineate guidelines for management of mutations like that for hyperekplexia in Patient 2) and for interpretation of polygenic change that is optimized by clinical genetic/syndromology experience (e.g., suggesting acetazolamide therapy for Patient 1 and explaining arthrogryposis in Patient 2)."
    >>> for s in qsummarise(question,text,n=3):
    ...   print(s[0] + "," + round(s[1],4))
    (0, 229) 1.0
    (230, 449) 0.9018
    (618, 1057) 0.9252
"""
    # global tfidf #, pca
    # Obtain the question and sentence features
    sentences = sent_tokenize(text)

    # # -- Uncomment this code to use tfidf of cluster information
    # sentence_words = [my_tokenize(s) for s in sentences+[question]]
    # sentence_clusters = []
    # for s in sentence_words:
    #     sentence_clusters.append([c for c in word2vec.clusters(s) if c])

    # tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    # tfidf.fit(sentence_clusters)
    # tfidf_question = tfidf.transform(sentence_clusters[-1:])
    # tfidf_sentences = tfidf.transform(sentence_clusters[:-1])
    # # --

    # -- Uncomment this code to use tfidf of words
    tfidf = TfidfVectorizer(stop_words="english",
                            lowercase=True)
    tfidf.fit(sentences+[question])

    tfidf_question = tfidf.transform([question])
    tfidf_sentences = tfidf.transform(sentences)
    # --

    # pca_question = pca.transform(tfidf_question)
    # pca_sentences = pca.transform(tfidf_sentences)


    # Score each sentence
    distances = pairwise_distances(tfidf_question,
                                   tfidf_sentences,
                                   'cosine')[0,:]
    scores = [(distances[i],i) for i in range(len(sentences))]

    # Obtain the top n sentences
    scores.sort()

    summary = scores[:n]

    # Find the character offsets
    offsets = []
    begin = 0
    for s in sentences:
        b = text.find(s,begin)
        assert(b >= 0)
        offsets.append((b,b+len(s)))
        begin = offsets[-1][1]

    # Return the results
    summary.sort(key=lambda x: x[1])
    #summary.sort(cmp = lambda x,y: cmp(x[1],y[1]))
    return [(offsets[i],score) for (score,i) in summary]

def bsummarise(question,text,n=3):
    """Summarise a document.

    Return the list of spans of text that best summarise a document,
    together with the span scores. The question and the text are strings.

    Example
    >>> question = "What is the best treatment for migraines?"
    >>> text = "Exome results are reported for two patients with connective tissue dysplasia, one refining a clinical diagnosis of Ehlers-Danlos to Marfan syndrome, the other suggesting arthrogryposis derived from maternofetal Stickler syndrome. Patient 1 had mutations in transthyretin (TTR), fibrillin (FBN1), and a calcium channel (CACNA1A) gene suggesting diagnoses of transthyretin amyloidosis, Marfan syndrome, and familial hemiplegic migraines, respectively. Patient 2 presented with arthrogryposis that was correlated with his mother's habitus and arthritis once COL2A1 mutations suggestive of Stickler syndrome were defined. Although DNA results often defy prediction by the best of clinicians, these patients illustrate needs for ongoing clinical scholarship (e.g., to delineate guidelines for management of mutations like that for hyperekplexia in Patient 2) and for interpretation of polygenic change that is optimized by clinical genetic/syndromology experience (e.g., suggesting acetazolamide therapy for Patient 1 and explaining arthrogryposis in Patient 2)."
    >>> for s in bsummarise(question,text,n=3):
    ...   print(s[0] + "," + round(s[1],4))
    (0, 229) 0.7927
    (230, 449) 0.7706
    (618, 1057) 1.0
    """

    # Obtain the sentence features
    sentences = sent_tokenize(text)
    tfidf = TfidfVectorizer(stop_words="english",
                            lowercase=True)
    features = tfidf.fit_transform(sentences)


    # lsa = TruncatedSVD(n_components=50)
    # features = lsa.fit_transform(tfidf_features)

    # Score each sentence
    scores = []
    for i in range(len(sentences)):
        f = features.getrow(i).toarray()[0,:]
        scores.append((f.sum(),i))
    maximum = np.max(scores)

    # Obtain the top n sentences
    scores.sort()
    summary = scores[-n:]

    # Find the character offsets
    offsets = []
    begin = 0
    for s in sentences:
        b = text.find(s,begin)
        assert(b >= 0)
        offsets.append((b,b+len(s)))
        begin = offsets[-1][1]

    # Return the results
    summary.sort(cmp = lambda x,y: cmp(x[1],y[1]))
    return [(offsets[i],np.max((0,score/maximum))) for (score,i) in summary]

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import sys
    sys.exit()

#    evaluateSummaries(bsummarise) # baseline
#    evaluateSummaries(qsummarise) # cosine tf.idf
#    evaluateSummaries(vsummarise) # cosine word2vec
#    evaluateSummaries(rsummarise) # regression-based summarisation
    evaluateSummaries(summarise) # default summarisation

