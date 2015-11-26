---.txt---
dict.txt: vocabulary for feature extraction
chi_n.txt: Chinese stopwords separated by '\n'
chi_,.txt: Chinese stopwords separated by ','
chi_,.backup: same as chi_,.txt
features_10000.txt: 10000 most common words from 26 data files
features_5000.txt: 5000 most common words from e-1 data files
jieba_cut_result.txt: jieba cut result of e-1.json
stanford_cut_result.txt: stanford cut result of ?
extraction.txt: e-1.json text content all together
jieba_pw_compareResult.txt: RT
output.txt: ? cut result(very large)
test*.txt: test text and its cutting result

---.py---
read.py: getText from .json file
bow.py: bag of words for jieba
tfidf.py: tfidf function
tfidf_cmd.py: run tfidf from command line
classify.py: classify based on bow

addstop.py: add stop word to three chi*.txt files
test.py: pwCount, jiebaCount, tfidf function
t.py: test
word2vec.py: word2vec using gensim


---.other---
jieba.model: Word2Vec model generated using jieba cutting

