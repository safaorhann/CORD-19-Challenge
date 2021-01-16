import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import stats
import researchpy as rp
from wordcloud import WordCloud, STOPWORDS 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


columns = ['cord_uid', 'who_covidence_id', 'sha', 'title', 'license', 'publish_time', 'journal', 'authors', 'url' ,'pdf_json_files']
path = 'C:/Users/HP/Desktop/arc/cord_19_embeddings/metadata.csv'
meta_df = pd.read_csv(path, usecols = columns, low_memory = False, dtype = {'who_covidence_id': str, 'pdf_json_files':str})


print(meta_df.head())
print(meta_df.info())
print(meta_df.isnull().sum())
print("number of paper that has title out of " + str(len(meta_df)) + " : " + str((len(meta_df)-(meta_df.title.isnull().sum()))))

print("number of distinct paper id : " + str(meta_df.sha.nunique()))
print("number of distinct title : " + str(meta_df.title.nunique()))


# articles with null title
print(meta_df[meta_df['title'].isnull()]['title'])


all_json = os.listdir('C:/Users/HP/Desktop/arc/document_parses/pdf_json')
path = 'C:/Users/HP/Desktop/arc/document_parses/pdf_json'
print(len(all_json))



import random
random_papers = random.sample(range(128915), 40000)
raw_json = []
#meta_list = []
for i in random_papers:
	with open(os.path.join(path, all_json[i])) as js:
		j = json.load(js)
		paper_id = j['paper_id']
		title = j['metadata']['title']
		try:
			abstract = j['abstract'][0]['text']
		except:
			abstract = ""
		full_text = ""
		bib_entries = []
		for text in j['body_text']:
			full_text += text['text']
			for csp in text['cite_spans']:
				try:
					title = j['bib_entries'][csp['ref_id']]['title']
					bib_entries.append(title)
				except:
					pass
		raw_json.append([paper_id, title, abstract, full_text, bib_entries])
		#meta_data = meta_df.loc[meta_df['sha'] == paper_id]
		#meta_list.append(meta_data)



df=pd.DataFrame(raw_json,columns=['paper_id','title','abstract','body','bib_entries']) 
#print(df['list_authors'])
#df.to_csv('new.csv')
#metadf = pd.DataFrame(meta_list, columns=columns)
meta_df.rename(columns = {'sha':'paper_id'}, inplace = True)
df = pd.merge(df, meta_df, how='inner', on= 'paper_id')
df.head()


def lower_case(input_str):
    input_str = input_str.lower()
    return input_str


df['body'] = df['body'].apply(lambda x: lower_case(x))
df['abstract'] = df['abstract'].apply(lambda x: lower_case(x))
df['body'].head()


incubation = df[df['body'].str.contains('incubation')]
incubation.head()


all_incubation_paragraph=[]
for text in incubation['body'].values:
    for paragraph in text.split('. '):
        if 'incubation' in paragraph:
            all_incubation_paragraph.append(paragraph)

print("len incubation paragraph: " + str(len(all_incubation_paragraph)))


virus = df[df['body'].str.contains('virus')]
virus.head()


all_virus_paragraph=[]
for text in virus['body'].values:
    for paragraph in text.split('. '):
        if 'virus' in paragraph:
            all_virus_paragraph.append(paragraph)

print("len virus paragraph: " + str(len(all_virus_paragraph)))


days_incubation=[]
for t in all_incubation_paragraph:
    day=re.findall(r"\d{1,2} day", t)
    if (len(day)==1):
        days_incubation.append(day[0].split(" "))
        
days_incubation_1=[]

for d in days_incubation:
    days_incubation_1.append(float(d[0]))

print("len days incubation: " + str(len(days_incubation_1)))

plt.xlabel("days incubation")
plt.hist(days_incubation_1)
plt.show()


print("mean incubation: " + str(np.mean(days_incubation_1)))


transmission = df[df['body'].str.contains('transmission')]
all_transmission_paragraph=[]
for text in transmission['body'].values:
    for paragraph in text.split('. '):
        if 'transmission' in paragraph:
            all_transmission_paragraph.append(paragraph)


print("len all_transmission_paragraph: " + str(len(all_transmission_paragraph)))

feet_transmission=[]
for t in all_transmission_paragraph:
    feet = re.findall(r"\d{1,2} feet", t)
    if len(feet)==1:
        feet_transmission.append(feet)

feet_transmission_1=[]
for d in feet_transmission:
    feet_transmission_1.append(float(d[0].split(' ')[0]))
    
print("len feet_transmission: " + str(len(feet_transmission_1)))
print("mean feet_transmission: " + str(np.mean(feet_transmission_1)))

plt.hist(feet_transmission_1)
plt.show()


#df['word_count'] = df['full_text'].apply(lambda x: len(str(x).split(" ")))
#df['char_count'] = df['full_text'].str.len() ## this also includes spaces
#print(df.head())

df['abstract_word_count'] = df['abstract'].apply(lambda x: len(x.strip().split()))  # word count in abstract
df['body_word_count'] = df['body'].apply(lambda x: len(x.strip().split()))  # word count in body
df['body_unique_words']=df['body'].apply(lambda x:len(set(str(x).split())))  # number of unique words in body
df.head()


df['abstract'].describe(include='all')
df['body'].describe(include='all')

#df.dropna(inplace=True)
df.info()

sns.displot(df['body_word_count'])
df['body_word_count'].describe()


sns.displot(df['body_unique_words'])
df['body_unique_words'].describe()


df[['abstract_word_count', 'body_word_count']].plot(kind='box', title='Boxplot of Word Count', figsize=(10,6))
plt.show()


def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

df['avg_word'] = df['body'].apply(lambda x: avg_word(x))
print(df[['body','avg_word']].head())


#stop is a list of all stop words in english
stop = stopwords.words('english')

#calculating number of stop words presents in body text of each paper
df['stopwords'] = df['body'].apply(lambda x: len([x for x in x.split() if x in stop]))
df[['body','stopwords']].head()

df['numerics'] = df['body'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['body','numerics']].head()


#Removing all digits in the text
df['body'] = df['body'].apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
df['body'].head()


#Removing punctuations
df['body'] = df['body'].str.replace('[^\w\s]','')
df['body'].head()


#remove stopwords
df['body'] = df['body'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['body'].head()


df.to_csv('metadata_paper_combined.csv', index= False)
df = pd.read_csv('C:/Users/HP/Desktop/arc/metadata_paper_combined.csv')

df['numerics'].median()
df['numerics'].min()
df['numerics'].max()
df['numerics'].mean()


norm_data = df['numerics']
print("Standard deviation of Numrics is: ",end="")  
print(norm_data.std())

print ("The varianve of the numeric is: ", end="")
print(norm_data.var())


# Deferencial
print(df['body_word_count'].min())
print (df['body_word_count'].max())

#infer
print(df['body_word_count'].median())
print(df['body_word_count'].mean())

norm_data = df['body_word_count']
print("Standard deviation of body word count is: ",end="")  
print(norm_data.std())

print ("The varianve of the body word count is: ", end="")
print(norm_data.var())


diff = df.body_word_count - df.body_unique_words
stats.probplot(diff, plot = plt)

stats.ttest_ind(df.body_word_count, df.body_unique_words)

rp.summary_cont(df.groupby('publish_time')['body_word_count'])

df.avg_word.mean()

df[['body_word_count', 'numerics']].plot(kind='box')
plt.show()

df['publish_time'].corr(df['body_word_count'], method = 'spearman')
df['publish_time'].corr(df['numerics'], method = 'spearman')
df['publish_time'].corr(df['stopwords'], method = 'spearman')
df['abstract_word_count'].corr(df['body_word_count'], method = 'spearman')
df['body_word_count'].corr(df['numerics'], method = 'spearman')
df['body_word_count'].corr(df['stopwords'], method = 'spearman')
df.stopwords.mean()
df.numerics.mean()
df.numerics.mode()


word_list = ['cell','infection', 'study', 'sample','virus', 'covid', 'sars', 'infection','pandemic','patient','data','model','transmission','effect','syndrome','protein','research','disease','change','mutation','vaccine','compared','syndrome','clinical','drug','viral','future','case','detection','health','public']
count = []
for i in word_list:
  count.append(tokenizer.word_counts[i])

print(count)

fig = plt.figure(figsize = (27, 8)) 
  
plt.bar(word_list, count)
plt.xlabel("words") 
plt.ylabel("frequencies") 
plt.title("word count plot") 
plt.show()


bodytext = []
for index, row in df.iterrows():
  bodytext.append(row['body'])


comment_words = '' 
stopwords = set(STOPWORDS) 
for val in bodytext[:2000]: 
    val = str(val) 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


comment_words = '' 
for val in bodytext[2000:4000]: 
    val = str(val) 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 



comment_words = '' 
for val in bodytext[20000:22000]: 
    val = str(val) 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 



comment_words = '' 
for val in bodytext[30000:32000]: 
    val = str(val) 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


comment_words = '' 
for val in df.abstract: 
    val = str(val) 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


comment_words = '' 

for val in df.authors: 
    val = str(val) 
    tokens = val.split()       
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


df['body_nonnumeric'] = df['body'].str.replace('\d+', '').astype(str)

bodytext = []
for index, row in df.iterrows():
  bodytext.append(row['body_nonnumeric'])


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(bodytext)

word_index = tokenizer.word_index
print(len(word_index))

seq = tokenizer.texts_to_sequences(bodytext[:20000])
seq1 = tokenizer.texts_to_sequences(bodytext[20000:])


max_length = 500
trunc_type='post'
padding_type='post'

padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
padded1 = pad_sequences(seq1, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = KMeans(n_clusters=2,init='k-means++')

ypred = model.fit_predict(padded)
plt.figure(figsize=(10,10))
plt.scatter(padded[ypred==0, 0], padded[ypred==0, 1], s =100, c = 'blue', label='cluster 1')
plt.scatter(padded[ypred==1, 0], padded[ypred==1, 1], s =100, c = 'red', label='cluster 2')
plt.show()


ypred = model.fit_predict(padded1)

plt.figure(figsize=(10,10))
plt.scatter(padded1[ypred==0, 0], padded1[ypred==0, 1], s =100, c = 'blue', label='cluster 1')
plt.scatter(padded1[ypred==1, 0], padded1[ypred==1, 1], s =100, c = 'red', label='cluster 2')
#plt.scatter(padded1[ypred==2, 0], padded1[ypred==2, 1], s =100, c = 'green', label='cluster 3')
plt.show()


titletext = df[df['title_x'].notnull()]['title_x']
token = Tokenizer(oov_token="<OOV>")
token.fit_on_texts(titletext)


titseq = tokenizer.texts_to_sequences(titletext)
paddedtitle = pad_sequences(titseq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

modeltitle = KMeans(n_clusters=2,init='k-means++')
ypred = model.fit_predict(paddedtitle)

plt.figure(figsize=(10,10))
plt.scatter(paddedtitle[ypred==0, 0], paddedtitle[ypred==0, 1], s =100, c = 'blue', label='cluster 1')
plt.scatter(paddedtitle[ypred==1, 0], paddedtitle[ypred==1, 1], s =100, c = 'red', label='cluster 2')
#plt.scatter(paddedtitle[ypred==2, 0], paddedtitle[ypred==2, 1], s =100, c = 'green', label='cluster 3')
plt.show()

abstext = df[df['abstract'].notnull()]['abstract']
tokenabs = Tokenizer(oov_token="<OOV>")
token.fit_on_texts(abstext)

absseq = tokenizer.texts_to_sequences(titletext)
paddedabs = pad_sequences(absseq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

modelabs = KMeans(n_clusters=2,init='k-means++')
ypred = model.fit_predict(paddedabs)

plt.figure(figsize=(10,10))
plt.scatter(paddedabs[ypred==0, 0], paddedabs[ypred==0, 1], s =100, c = 'blue', label='cluster 1')
plt.scatter(paddedabs[ypred==1, 0], paddedabs[ypred==1, 1], s =100, c = 'red', label='cluster 2')
plt.scatter(paddedabs[ypred==2, 0], paddedabs[ypred==2, 1], s =100, c = 'green', label='cluster 3')

plt.show()

x = df[["abstract_word_count","body_word_count","body_unique_words","avg_word","numerics"]].values
y = df['stopwords'].values

X_train, X_test, y_train, y_test = train_test_split(x, y)

reg = LinearRegression().fit(X_train, y_train)
reg.score(X_test, y_test)
reg.coef_
reg.intercept_


bcount = df[['body_word_count', 'body_unique_words']].values

modelbcount = KMeans(n_clusters=3,init='k-means++')
ypred = model.fit_predict(bcount)

plt.figure(figsize=(10,10))
plt.scatter(bcount[ypred==0, 0], bcount[ypred==0, 1], s =100, c = 'blue', label='cluster 1')
plt.scatter(bcount[ypred==1, 0], bcount[ypred==1, 1], s =100, c = 'red', label='cluster 2')
plt.scatter(bcount[ypred==2, 0], bcount[ypred==2, 1], s =100, c = 'green', label='cluster 3')
plt.show()













