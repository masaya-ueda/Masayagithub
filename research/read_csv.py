import pandas as pd
import itertools
import numpy as np


def read_csv_space(csv_path):
    col_names = [ 'c{0:02d}'.format(i) for i in range(185) ]
    data = pd.read_csv(csv_path, sep=" ",header=None,engine='python', names=col_names).values.tolist()
    data=list(itertools.chain.from_iterable(data))
    All_Font_list=[]
    for i in range(len(data)):
        if data[i]==data[i]:
            All_Font_list.append(data[i])
        else:
            pass

    All_Font_list = list(filter(None, All_Font_list))
    for j in  range (len(All_Font_list)):
        All_Font_list[j]=str(All_Font_list[j]).strip()
    
    return All_Font_list

def read_csv_comma(csv_path):
    col_names = [ 'c{0:02d}'.format(i) for i in range(185) ]
    data = pd.read_csv(csv_path, sep=",",header=None,engine='python', names=col_names).values.tolist()
    data=list(itertools.chain.from_iterable(data))
    All_Font_list=[]
    for i in range(len(data)):
        if data[i]==data[i]:
            All_Font_list.append(data[i])
        else:
            pass

    All_Font_list = list(filter(None, All_Font_list))
    for j in  range (len(All_Font_list)):
        All_Font_list[j]=str(All_Font_list[j]).strip()
    return All_Font_list

def read_csv(csv_path):
    col_names = [ 'c{0:02d}'.format(i) for i in range(186) ]
    data = pd.read_csv(csv_path, sep=" ",header=None,engine='python', names=col_names).values.tolist()
    
    font_tag_list=[]
    for i in range(len(data)):
        data[i] = list(filter(None, data[i]))
        font_list = []
        for j in range(len(data[i])):
            if data[i][j] == data[i][j]:
                font_list.append(data[i][j])
            else:
                pass
        font_tag_list.append(font_list)
    return font_tag_list