import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
nltk.download('punkt')

import re

paragraph="""To get the Unique you, there is a big battle. There are four things you need to have, to win that battle. The first is to set the goal, the second is to acquire the knowledge continuously, third is work hard with devotion and fourth is perseverance. If you have these four tools then you will definitely become UNIQUE YOU.

Being Unique will require excellence. Excellence is self-imposed, self- directory life long process.

Excellence is not by accident. It is process, where an individual, organisation or nation continuously strive to better oneself. The performance standards are set by themselves, they work on their dreams with focus and are prepared to take calculated results and do not get deterred by failures as they move towards their dreams. Then, they step up their dreams as they tend to their potential, in the process they increase their potential, and this is an unending life cycle phenomenon. They are not in competition with anybody else, but themselves. That is the culture of Excellence.

Students ought to think, and think well. They should do everything to build up a new state of India which would be everybody’s pride.

I would like to ask yourself, what you would like to remembered for? You should write this on a page, and that page will be most important page in book of human history. And you will be remembered for creating that page in the history of nation."""


text=re.sub(r'\[[0-9]*\]', " ",paragraph)
text=re.sub(r'\s+',' ',text)
text=text.lower()
text=re.sub(r'\d',' ',text)

sentences=nltk.sent_tokenize(text)

sentences= [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i]= [word for word in sentences[i] if word not in stopwords.words('english')]
    
model= Word2Vec(sentences,min_count=1)

words= model.wv.key_to_index 

vector= model.wv['unique']

similar=model.wv.most_similar('unique')