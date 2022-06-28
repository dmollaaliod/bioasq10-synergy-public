import nltk
from nltk.corpus import stopwords

porter = nltk.PorterStemmer()

def my_tokenize(string):
    """Return the list of tokens.
    >>> my_tokenize("This is a sentence. This is another sentence.")
    ['sentence', 'another', 'sentence']
    """
#    return [porter.stem(w.lower()) 
    return [w.lower() 
            for s in nltk.sent_tokenize(string) 
            for w in nltk.word_tokenize(s)
            if w.lower() not in stopwords.words('english') and
               w not in [',','.',';','(',')','"',"'",'=',':','%','[',']']]
 
if __name__ == "__main__":
    import doctest
    doctest.testmod()
