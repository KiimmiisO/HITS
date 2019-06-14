import deepcut
from pythainlp import word_tokenize

f_in = open("../data/data/tweet_hub.txt", "w", encoding='UTF-8')
for text in f_in.read():
       print(text)
       print(text.replace(' ', ''))
       proc = word_tokenize(text.replace(' ', ''), engine='deepcut')
       print(proc)

f = open("word_wrapping.txt", "w", encoding='UTF-8')
for i in proc:
    f.write("{} ".format(i))
f.write("\n")
f.close()
