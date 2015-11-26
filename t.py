import re
import sys
sys.path.append("./data")

f=open('extraction.txt')
print f.read()
f.close()

#infile=open('chi_,.txt','r')
#of=open('chi_n.txt','w')#'w')
#txt=infile.read().split('.')
#txt=re.sub(',',"\n",txt[0])
#print txt.__class__
#for x in range(10,10000):
#    of.write(","+str(x))
#txt=",01,02,03,04,05,06,07,08,09"
#of.write(txt)

