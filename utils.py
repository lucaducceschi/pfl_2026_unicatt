from nltk import word_tokenize, sent_tokenize
def infile_outsents_comp(p):
    return [word_tokenize(sent.lower()) for sent in sent_tokenize(open(p, encoding='utf-8-sig').read())]


from collections import defaultdict
from string import punctuation
punctuation += "”“’--"
from nltk.corpus import stopwords
en_stops = stopwords.words('english')

def get_bigrams_freqs(corpus, threshold= 5):
    # populate the dictionary with bigrams frequencies, where bigrams are keys and their value is the frequency
    # out = {}
    out = defaultdict(int)
    for text in corpus:
        for w1,w2 in zip(text, text[1:]):
            if w1 not in punctuation and w2 not in punctuation and w1 not in en_stops and w2 not in en_stops:
                bigram = f'{w1}_{w2}'
                # if bigram in out:
                #     out[bigram] += 1
                # else:
                #     out[bigram] = 1
                out[bigram] +=1
    sorted_freqs = sorted(out.items(), key=lambda x : x[1], reverse=True )
    return [t for t in sorted_freqs if t[1] >= threshold]
