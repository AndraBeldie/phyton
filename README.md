import pandas as pd
amazon=pd.read_excel('C:/Users/Danu/Desktop/ proiect analiza text/BD_Amazon.xlsx')
print (amazon)
#Afisarea numelor coloanelor
columnNames = list(amazon.head())
print(columnNames)
#Afiseaza tipul variabilelor
amazon.dtypes
amazon.info()
df_excel=amazon
df_excel['a-section']  = df_excel['a-section'].str.lower()

df_excel['a-section'] = df_excel['a-section'].astype(str)
import nltk
import re
import sys

from nltk.corpus import words
words = set(nltk.corpus.words.words())

def remove_noneng(amazon):  
    text = [w for w in nltk.wordpunct_tokenize(amazon) if w.lower() in words or not w.isalpha()]
    return text

df_excel['a-section'] = df_excel['a-section'].astype(str)    
df_excel['nonenglish'] = df_excel['a-section'].apply(lambda x: remove_noneng(x))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 

df_excel['comments_lemm'] = df_excel['a-section'].apply(lambda x: lemmatizer.lemmatize(x)) 
print(lemmatizer.lemmatize("cars"))
#stopword
import nltk
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df_excel['tokenized_sentences_nonstop'] = df_excel['a-section'].apply(lambda x: remove_stopwords(x))
df_excel.head(10)

df_excel['comments_processed'] = df_excel['a-section'].apply(lambda x: preprocess(x)) 


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>!,')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

df_excel['comments_processed'] = df_excel['review-text'].apply(lambda x: preprocess(x))
df_excel['review'] = df_excel['review-text']
df_excel['comments_processed'] = df_excel.review.apply(lambda x: word_tokenize(x))



from nltk.stem import WordNetLemmatizer
import CorrectPythonPackage.token as token2


lemmatizer = WordNetLemmatizer() 
print(lemmatizer.lemmatize(df_excel.review[1]))

counter = 0
lemm_tokens = []
for i in tokens_w:
    lemm_tokens.append(lemmatizer.lemmatize(i))
print(lemm_tokens)



# remove stopwords from the text

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

wordsFiltered = []
for w in lemm_tokens:
    if w not in stopWords:
        wordsFiltered.append(w)
print(wordsFiltered) 

df_excel['review'].apply(nltk.word_tokenize)
df_excel['tokenized_sentences'] = df_excel['review'].apply(nltk.word_tokenize)

 #from nltk.tokenize import RegexpTokenizer
 #tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
#df_excel['tokenized_sentences'] = df_excel['a-section'].apply(lambda x: tokenizer.tokenize(x))



df_excel['tokenized_sentences_nltk'] = df_excel['a-section'].apply(nltk.word_tokenize) # it take some times
df_excel['tokenized_sentences_naive'] = df_excel['a-section'].apply(lambda s: s.split(' .'))



#frecvente

# function to plot most frequent terms
y=df_excel.tokenized_sentences_naive
import matplotlib.pyplot as plt
import sys
import operator
import argparse

def freq_words(y, terms = 30):
  all_words = ' '.join(text for text in y )
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
  
  # eliminare ""
 
        
        
  # cod bun
  import pandas as pd
import nltk
data = x
top_N =40
  
#if not necessary all lower
x = df_excel['a-section'].str.lower().str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(x)

word_dist = nltk.FreqDist(words)
print (word_dist)


rslt = pd.DataFrame(word_dist.most_common(top_N),
                    columns=['word', 'count'])
print(rslt)

rslt.plot(x ='word', y='count', kind='bar')

# plot stele / numele cartii
b= df_excel['Format'].str.lower().str.cat(sep=' ')
words1 = nltk.tokenize.word_tokenize(b)

word_dist1 = nltk.FreqDist(words1)
print (word_dist1)
rslt1 = pd.DataFrame(word_dist1.most_common(top_N),
                    columns=['word', 'count'])

print(rslt1)
rslt1.plot(x ='word', y='count', kind='bar')
rslt1.plot.pie(y='count',figsize=(5, 5),autopct='%1.1f%%', startangle=90)



