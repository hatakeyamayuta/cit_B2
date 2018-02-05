import requests
from bs4 import BeautifulSoup
url = 'http://www.gutenberg.org/files/11/11-h/11-h.htm'
html = requests.get(url)
souces = BeautifulSoup(html.text ,"html.parser")
   # obj = souce.a.strin
print(str(souces.find_all("p")))
with open('test1.txt','w') as f:
    f.write(str(souces.find_all("p")))
