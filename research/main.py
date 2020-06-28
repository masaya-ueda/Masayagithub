# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import statistics as st
import os

from newimage import crop, add_white, new_image
from read_csv import read_csv
from BagOfFeatures import *


#画像群からそれぞれの記述子を計算する関数。一応特徴点を検出し、その周りから抽出
def detectAndCompute(image_paths):
    # Detect, compute and return all features found on images
    sift=cv2.xfeatures2d.SIFT_create()
    descriptions=np.empty((0,128))
    i=0
    for image_path in image_paths:
        image_crop = crop(image_path)
        image_blank = add_white(image_crop)
        image_new = new_image(image_crop,image_blank)
        image_new = image_new.astype(np.uint8)

        file_name=os.path.basename(image_path)
        save_path=os.path.join('/home/masaya/research/real/image', file_name)

        if "_AA.png" in file_name:
            cv2.imwrite(save_path,image_crop)
        else:
            pass

        if image_new.shape != (0,0,3):
            kp[image_path], features[image_path]=sift.detectAndCompute(image_new,None)
            if features[image_path] is None:
                i=i+1
                print(image_path)
            else:
                descriptions=np.concatenate([descriptions,features[image_path]])
        else:
            print(image_path)
    if i>0:
        print(i)
        
    else:
        pass
        #print("PASS")
    return descriptions

if __name__== "__main__":

    codebookSize=15
    features={}
    hist={}
    bar={}
    bar_list={}
    bar_average={}
    errorbar={}
    kp={}
    tag1="ancient"
    tag2="modern"

    data_all_csv_path="/home/masaya/research/test/all_test.csv"
    #data_all_csv_path="/home/masaya/research/real/AllFont.csv"
    tag1_csv_path="/home/masaya/research/test/ancient_test.csv"
    #tag1_csv_path="/home/masaya/research/real/ancient.csv"
    tag2_csv_path="/home/masaya/research/test/modern_test.csv"
    #tag2_csv_path="/home/masaya/research/real/modern.csv"

    #　#　#　全フォントのCSVファイルを読み込み，画像からSIFT特徴量を抽出し，クラスタリングする　#  #  #

    #data_all = pd.read_csv(data_all_csv_path,header=None).values.tolist()
    #allfont_list=data_all[0]
    allfont_list=read_csv(data_all_csv_path)
    print(allfont_list)
    print(len(allfont_list))
    features["all"]=np.empty((0,128))
    for i in range(len(allfont_list)):
        Font_name=allfont_list[i]

        #全フォント画像のファイル名を獲得
        path=os.path.join('/home/masaya/ダウンロード/dataset/fontimage', Font_name)
        path=path+'*.png'
        files = glob.glob(path)

        #それぞれの画像からSIFT記述子を計算
        features[Font_name]=detectAndCompute(files)
        
        features["all"]=np.concatenate([features["all"],features[Font_name]])
    features["all"]=features["all"].astype(np.float32)
    print("---SIFT finished---")
    
    #全画像の特徴ベクトルをkmeansクラスタリングし，Visual Wordを決定 
    bof=BagOfFeatures(codebookSize)
    bof.train(features["all"])
    print("---clustering finished---")

#   #   #   全フォントの特徴点から全フォントの平均ヒストグラムを作成   #   #   #

    bar["all"]=[0 for _ in range(codebookSize)]

    #ヒストグラムの各次元の要素を格納する空のリストを作成
    bar_list["all"] = [[] for _ in range(codebookSize)]
    letter_list=["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ", "KK", "LL", "MM", "NN", "OO", "PP", "QQ", "RR", "SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ", "a", "b", "c", "d",
                 "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    for i in range(len(allfont_list)):
        #選ばれたフォントのパスを取得し画像を読み込む
        Font_name=str(allfont_list[i])
        bar[Font_name]=[0 for _ in range(codebookSize)]
        for letter in letter_list:
            path=os.path.join('/home/masaya/ダウンロード/dataset/fontimage', Font_name)
            path=path+'_'
            path=path+letter
            path=path+".png"
            #print(path)
            image_crop = crop(path)
            image_blank = add_white(image_crop)
            image_new = new_image(image_crop,image_blank)
            image_new = image_new.astype(np.uint8)

            if image_new.shape != (0,0,3):
                
                if features[path] is None:
                    print("---description is None----")
                    print(path)
                else:
                    features[path]=features[path].astype(np.float32)
                    #選ばれたフォントのヒストグラムを作成
                    histogram=list(map(lambda a:bof.makeHistgram(np.matrix(a)),features[path]))
                    histogram=list(map(lambda a:np.argmax(a),histogram))
                    #print(histogram)
                    #棒グラフのリスト形式に変換
                    bar[path]=[0 for _ in range(codebookSize)]
                    for i in range(len(histogram)):
                        for j in range(codebookSize):
                            if histogram[i]==j:
                                bar[path][j]+=1
                            else:
                                pass
                    #正規化
                    bar[path]=list(map(lambda x: x / sum(bar[path]), bar[path]))
                    #print(bar[path])
                    bar[Font_name]=[x + y for (x, y) in zip(bar[Font_name], bar[path])]

            else:
                print("---image is empty---")
                print(path)
                
        
        bar["all"]=[x + y for (x, y) in zip(bar["all"], bar[Font_name])]

        #標準偏差計算のためのリスト作成
        for i in range(codebookSize):
            bar_list["all"][i].append(bar[Font_name][i]) 

    #全フォントの平均ヒストグラムを作成
    bar_average["all"]=list(map(lambda x: x / len(allfont_list), bar["all"]))
    print(sum(bar_average["all"]))
    #標準偏差を計算しリストに格納
    errorbar["all"]=list(map(lambda x:st.pstdev(x),bar_list["all"]))
    #print(errorbar["all"])

    data = bar_average["all"]
    labels = list(range(1,codebookSize+1))
    fig = plt.figure()

    plt.bar(labels, data, width=1.0,
                color='gray',
                edgecolor='black',
                linewidth=1.0,
                yerr=errorbar["all"],
                ecolor='black',
                align='center')

    plt.xlabel("dimension")
    plt.ylabel('Number')
    plt.title('all_average')
    plt.ylim(0,20)
    #plt.show()

    #ヒストグラムの保存
    #plt.savefig("/home/masaya/research/test/average/average.png")
    plt.savefig("/home/masaya/research/real/average/15dimensions/average.png")
    print("---making all-average-histogram finished---")
    
#   #   #   比較フォント１の平均ヒストグラムを作成  #   #   #   

    bar[tag1]=[0 for _ in range(codebookSize)]
    #ヒストグラムの各次元の要素を格納する空のリストを作成
    bar_list[tag1] = [[] for _ in range(codebookSize)]

    #比較フォント１のファイル名のcsvファイルを読み込んでリストに格納
    data1 = pd.read_csv(tag1_csv_path,header=None).values.tolist()
    tag1_list=data1[0]
    print(tag1_list)
    print(len(tag1_list))
    
    for i in range(len(tag1_list)):
        Font_name=str(tag1_list[i])
        bar[tag1]=[x + y for (x, y) in zip(bar[tag1], bar[Font_name])]

        #標準偏差計算のためのリスト作成
        for i in range(len(bar[Font_name])):
            bar_list[tag1][i].append(bar[Font_name][i]) 
    
    #比較フォント１の平均ヒストグラムを作成して保存
    bar_average[tag1]=list(map(lambda x: x / len(tag1_list), bar[tag1]))
    print(sum(bar_average[tag1]))
    #標準偏差を計算しリストに格納
    errorbar[tag1]=list(map(lambda x:st.pstdev(x),bar_list[tag1]))
    #print(errorbar[tag1])

    data = bar_average[tag1]
    labels = list(range(1,codebookSize+1))
    fig = plt.figure()

    plt.bar(labels, data, width=1.0,
                color='red',
                edgecolor='black',
                linewidth=1.0,
                yerr=errorbar["ancient"],
                ecolor='black',
                align='center')

    plt.xlabel("dimension")
    plt.ylabel('Number')
    plt.title('ancient_average')
    plt.ylim(0,20)
    #plt.legend()

    #ヒストグラムの保存
    plt.savefig("/home/masaya/research/real/bar1/15dimensions/bar1.png")
    print("---making tag1-average-histogram finished---")

#   #   #   比較フォント2の平均ヒストグラムを作成  #   #   #

    bar[tag2]=[0 for _ in range(codebookSize)]
    #ヒストグラムの各次元の要素を格納する空のリストを作成
    bar_list[tag2] = [[] for _ in range(codebookSize)]

    #比較フォント2のファイル名のcsvファイルを読み込んでリストに格納
    data2 = pd.read_csv(tag2_csv_path,header=None).values.tolist()
    tag2_list=data2[0]
    print(tag2_list)
    print(len(tag2_list))

    #print(tag2_list)
    bar[tag2]=[0 for _ in range(codebookSize)]
    for i in range(len(tag2_list)):
        Font_name=str(tag2_list[i])
        bar[tag2]=[x + y for (x, y) in zip(bar[tag2], bar[Font_name])]

        #標準偏差計算のためのリスト作成
        for i in range(len(bar[Font_name])):
            bar_list[tag2][i].append(bar[Font_name][i])

    #比較フォント2の平均ヒストグラムを作成
    bar_average[tag2]=list(map(lambda x: x / len(tag2_list), bar[tag2]))
    print(sum(bar_average[tag2]))

    #標準偏差を計算しリストに格納
    errorbar[tag2]=list(map(lambda x:st.pstdev(x),bar_list[tag2]))
    #print(errorbar[tag2])
   
    data = bar_average[tag2]
    labels = list(range(1,codebookSize+1))
    fig = plt.figure()

    plt.bar(labels, data, width=1.0,
                color='blue',
                edgecolor='black',
                linewidth=1.0,
                yerr=errorbar["modern"],
                ecolor='black',
                align='center')

    plt.xlabel("dimension")
    plt.ylabel('Number')
    plt.title('modern_average')
    plt.ylim(0,20)
    #plt.legend()

    #ヒストグラムの保存
    plt.savefig("/home/masaya/research/real/bar2/15dimensions/bar2.png")
    print("---making tag2-average-histogram finished---")

#   #   #   平均ヒストグラムの差を取り棒グラフにして保存    #   #   #

    bar["comp1"]=[x - y for (x, y) in zip(bar_average[tag1], bar_average["all"])]
    bar["comp2"]=[x - y for (x, y) in zip(bar_average[tag2], bar_average["all"])]
    
    

    n_groups = codebookSize

    data_a = bar["comp1"]
    data_b = bar["comp2"]

    labels = list(range(1,codebookSize+1))
    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.50

    rects1 = ax.bar(index, data_a, bar_width,
                color='r',
                label='ancient',
                linewidth=[1 for _ in range(codebookSize)],
                edgecolor="black")

    rects2 = ax.bar(index + bar_width, data_b, bar_width,
                color='b',
                label='modern',
                linewidth=[1 for _ in range(codebookSize)],
                edgecolor="black")

    ax.set_xlabel("dimension")
    ax.set_ylabel('Number')
    ax.set_title('Comparison of feature')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(-10,10)
    ax.legend()

    fig.tight_layout()
    plt.savefig("/home/masaya/research/real/result/15dimensions/result.png")
    print("Done")
    plt.show()


#   #   #   特徴点の描画    #   #   #
    color = {}
    color[0] = (0, 0, 0) #黒
    color[1] = (0, 0, 255) #赤
    color[2] = (0, 255, 0) #緑
    color[3] = (255, 0, 0) #青
    color[4] = (0, 255, 255) #黃
    color[5] = (255, 255, 0) #水色
    color[6] = (255, 0, 255) #紫
    color[7] = (128, 128, 128) #灰色
    color[8] = (0, 0, 128) #茶
    color[9] = (128, 0, 0) #ネイビー
    color[10] = (0, 150,200) #ベージュ
    color[11] = (242, 204, 255) #ピンク
    color[12] = (0, 127, 255) #オレンジ
    color[13] = (240, 255, 255) #白
    color[14] = (255, 255, 204) #青緑
    #color[15] = (, , ) #
    #color[16] = (, , ) #
    #color[17] = (, , ) #
    #color[18] = (, , ) #
    #color[19] = (, , ) #

    for i in range(len(tag1_list)):
        Font_name=str(tag1_list[i])
        #全フォント画像のファイル名を獲得
        path=os.path.join('/home/masaya/ダウンロード/dataset/fontimage', Font_name)
        path=path+'*.png'
        files = glob.glob(path)
        
        for image_path in files:
            image_crop = crop(image_path)
            image_blank = add_white(image_crop)
            image_new = new_image(image_crop,image_blank)
            image_new = image_new.astype(np.uint8)
            if image_new.shape != (0,0,3):
                
                if features[image_path] is None:
                    print("---description is None----")
                    print(path)
                else:
                    features[image_path]=features[image_path].astype(np.float32)
                    #選ばれたフォントのヒストグラムを作成
                    histogram=list(map(lambda a:bof.makeHistgram(np.matrix(a)),features[image_path]))
                    histogram=list(map(lambda a:np.argmax(a),histogram))
                    #print(histogram)

                    for num in range(codebookSize):
                        kp_list=[]
                        kp_list=[i for i, x in enumerate(histogram) if x == num]
                        #print(image_path)
                        #print(kp_list)
                        if kp_list ==[]:
                            pass
                        else:
                            for k in range(len(kp_list)):
                                kp_fire =[]
                                kp_fire.append(kp[image_path][kp_list[k]])
                                image_new=cv2.drawKeypoints(image_new,kp_fire,None, color=color[num] ,flags=4)
                    file_name=os.path.basename(image_path)
                    path=os.path.join('/home/masaya/research/draw/ancient/15dimensions', file_name)
                    #print(path)
                    cv2.imwrite(path, image_new)


            else:
                print("---image is empty----")
                print(image_path)

    for i in range(len(tag2_list)):
        Font_name=str(tag2_list[i])
        #全フォント画像のファイル名を獲得
        path=os.path.join('/home/masaya/ダウンロード/dataset/fontimage', Font_name)
        path=path+'*.png'
        files = glob.glob(path)
        
        for image_path in files:
            image_crop = crop(image_path)
            image_blank = add_white(image_crop)
            image_new = new_image(image_crop,image_blank)
            image_new = image_new.astype(np.uint8)
            if image_new.shape != (0,0,3):
                
                if features[image_path] is None:
                    print("---description is None----")
                    print(path)
                else:
                    features[image_path]=features[image_path].astype(np.float32)
                    #選ばれたフォントのヒストグラムを作成
                    histogram=list(map(lambda a:bof.makeHistgram(np.matrix(a)),features[image_path]))
                    histogram=list(map(lambda a:np.argmax(a),histogram))
                    #print(histogram)

                    for num in range(codebookSize):
                        kp_list=[]
                        kp_list=[i for i, x in enumerate(histogram) if x == num]
                        #print(image_path)
                        #print(kp_list)
                        if kp_list ==[]:
                            pass
                        else:
                            for k in range(len(kp_list)):
                                kp_fire =[]
                                kp_fire.append(kp[image_path][kp_list[k]])
                                image_new=cv2.drawKeypoints(image_new,kp_fire,None, color=color[num] ,flags=4)
                    file_name=os.path.basename(image_path)
                    path=os.path.join('/home/masaya/research/draw/modern/15dimensions', file_name)
                    #print(path)
                    cv2.imwrite(path, image_new)


            else:
                print(image_path)