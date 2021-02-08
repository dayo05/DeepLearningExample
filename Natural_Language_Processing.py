en_text = "I'm learning Python with Intellij Pycham"

# Using spaCy version
import spacy

spacy_en = spacy.load("en_core_web_sm")


def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]


print(tokenize(en_text))


# Using nltk
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize


print(word_tokenize(en_text))


# Using split
print(en_text.split())


ko_text = '나는 인텔리제이의 파이참을 이용해서 머신러닝 공부를 하고 있다'


# Using split
print(ko_text.split())


'''
# Using Konlpy(Windows not supported)
from konlpy.tag import Mecab


tokenizer = Mecab()
print(tokenizer.morphs(ko_text))
'''
