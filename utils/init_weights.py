import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def class_per_dir(dir):
    mydict = {}
    x=[]
    y=[]
    for item in os.listdir(dir): # item:lollipop
        if os.path.isdir(dir): # 判断是否是文件夹
            path=os.path.join(dir,item) #path='../data/multi/sketch/lollipop'
            if(item=='.DS_Store'):
                continue
            temp=0
            for itemfile in os.listdir(path): # itemfile: sketch_178_000104
                filename=os.path.join(path,itemfile) #filename='../data/multi/sketch/lollipop/sketch_178_000104.jpg'
                if os.path.isfile(filename):
                    temp+=1;
                index=itemfile.split('_')[1]
            mydict[index,item]=temp
    jmq=sorted(mydict.items(), key = lambda kv:(kv[1], kv[0]))
    for key,value in jmq:
        x.append(key[0])
        y.append(value)
    x=np.array(x)
    y=np.array(y)
    plt.bar(x,y)
    plt.xticks([])
    plt.xlabel("class index")
    plt.ylabel("num")
    name=dir.split('/')[3]
    plt.title(name)
    plt.show()
    return jmq
#类别标号，类别名字，数量是多少
sketch=class_per_dir('../data/multi/sketch')
real=class_per_dir('../data/multi/real')
print(sketch)
print(real)

