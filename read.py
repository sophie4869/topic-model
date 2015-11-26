#_*_ coding: utf-8 _*_
import json
import sys
sys.path.append("/home/tingyubi/20w/data/")

# getText
# params: @prefix: prefix in data file name prefix-X.json
#         @begin/end: range for X in prefix-X.json
#         @quality: return result has quality(context)>=X
#         @context: 'text'/'titile'/'ad-slots'
# return: array of quality-context, content-context
def getText(path="/home/tingyubi/20w/data/",prefix='extraction-',begin=1,end=26,quality = 2, context = 'text'):
    d=[]
    for i in range(begin,end+1):
        fp=path+prefix+str(i)+'.json'
        d+=json.load(open(fp))
    q=[d[i]['quality']['text'] for i in range(len(d))]
    txt=[d[i]['content'][context] for i in range(len(d))]
    qret=[]
    tret=[]
    for qi,ti in zip(q,txt):
        if qi==quality:
            qret.append(qi)
            tret.append(ti)
    return qret, tret


# getData
# params: @source: extraction-X.json, X=0 for all
#         @quality: return result has quality(context)>=X
#         @context: 'text'/'titile'/'ad-slots'
#         @write: write to extraction.txt or not
# return: quality-context; dict, array of content-context
def getData(source = 0, quality = 2, context = 'text', write = False):
    d=[]
    if source:
        d+=json.load(open('extraction-'+str(source)+'.json'))
    else:
        for i in range(1,27):
            fp='extraction-'+str(i)+'.json'
            d+=json.load(open(fp))
    
    q=[d[i]['quality']['text'] for i in range(len(d))]
    t=[d[i]['content']['title'] for i in range(len(d))]
    txt=[d[i]['content'][context] for i in range(len(d))]
    qret=[]
    tret={}# problem: duplicated titles
    text=[]
    for qi,ti in zip(q,zip(t,txt)):#t.items()):
        if qi==quality:
            qret.append(qi)
            text.append(ti[1])
            if(ti[0]) == "":
                tret[ti[1][:6]]=ti[1]
            else:
                tret[ti[0]]=ti[1]#.append(ti.encode('utf-8'))
    if(write):
        of = open('extraction.txt','w')
        for i in range(len(tret)):
            of.write(tret[i]+"\n")
        of.close()
    return qret, tret, text


