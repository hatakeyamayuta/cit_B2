from collections import Counter
from numpy.random import *
import re
sum_w2 = 0
sum_w3 = 0
words = []
words2 = []
words2_number = []
words2_con = []
words3 = []
words3_number = []
words3_con = []
rand_words = []
rand_words2 = []
rand_words3 = []
def essential_figure(par):
    par *=10000
    par=int(par)
    par=float(par)/100
    return par
with open('test.txt','r') as f:
    obj=f.read()

obj = obj.lower()
obj = re.sub(re.compile("[—!'\'\",’“”‘\/:-@.[`{~*)(+0123456789]"), '', obj)
obj = re.sub(re.compile("\n"),'',obj)
sum_words = len(obj)
sum_spel = len(obj.split(" "))


for word in Counter(obj).most_common():
    with open('word_count.txt','a')as w:
        w.write(str(word))
        par = (word[1])/(sum_words)
        par=essential_figure(par)
        words.append(word[0])
        rand_words.append(par)
        w.write('  {0}%'.format(str(par))+'\n')

for i in range (ord(' '),ord('z')):
    if i>ord(' ') and i<ord('a'):
        continue
    for k in range (ord(' '),ord('z')):
        if k>ord(' ') and k<ord('a')or obj.count('{0}{1}'.format(chr(i),chr(k)))==0:
            continue
        #print('{0}{1}'.format(chr(i),chr(k)),obj.count('{0}{1}'.format(chr(i),chr(k))))
        words2.append('{0}{1}'.format(chr(i),chr(k)))
        words2_number.append(obj.count('{0}{1}'.format(chr(i),chr(k))))
        sum_w2 += int(obj.count('{0}{1}'.format(chr(i),chr(k))))
words2_con=list(zip(words2,words2_number))
words2_con.sort(key=lambda x: x[1],reverse=True)
for word2 in words2_con:
    par = word2[1]/sum_w2
    par = essential_figure(par)
    rand_words2.append(par)
    with open('word2_count.txt','a')as w2:
        w2.write(str(word2)+' '+str(par)+'%\n')

for n in range (ord(' '),ord('z')):
    if n>ord(' ') and n<ord('a'):
        continue
    for i in range (ord(' '),ord('z')):
        if i>ord(' ') and i<ord('a'):
            continue
        for k in range (ord(' '),ord('z')):
            if k>ord(' ') and k<ord('a') or obj.count('{0}{1}{2}'.format(chr(n),chr(i),chr(k)))==0:
                continue
           #print('{0}{1}{2}'.format(chr(n),chr(i),chr(k)),obj.count('{0}{1}{2}'.format(chr(n),chr(i),chr(k))))
            words3.append('{0}{1}{2}'.format(chr(n),chr(i),chr(k)))
            words3_number.append(obj.count('{0}{1}{2}'.format(chr(n),chr(i),chr(k))))
            sum_w3 += obj.count('{0}{1}{2}'.format(chr(n),chr(i),chr(k)))
words3_con=list(zip(words3,words3_number))
words3_con.sort(key=lambda x: x[1],reverse=True)
for word3 in words3_con:
    par = word3[1]/sum_w3
    par = essential_figure(par)
    rand_words3.append(par)
    with open('word3_count.txt','a')as w3:
        w3.write(str(word3)+' '+str(par)+'%\n')

print(' '.join(choice(words,50,rand_words)))
print(' '.join(choice(words2,50,rand_words2)))
print(' '.join(choice(words3,50,rand_words3)))

for spel in Counter(obj.split(" ")).most_common(50):
    #print(spel)
    with open('spel_count.txt','a')as s:
        s.write(str(spel))
        par = (spel[1])/(sum_spel)
        par = essential_figure(par)
        s.write('  {0}%'.format(str(par)))
        s.write('\n')
