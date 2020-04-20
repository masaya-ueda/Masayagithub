# -*- coding: utf-8 -*-
"""task2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o_N-zAIWRQE_HvrRx6r3OPchSRVAR08-
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

!gunzip "/content/drive/My Drive/Colab Notebooks/datasets/train.tar.gz"
!gunzip "/content/drive/My Drive/Colab Notebooks/datasets/test.tar.gz"
!gunzip "/content/drive/My Drive/Colab Notebooks/datasets/val.tar.gz"

!tar xvf "/content/drive/My Drive/Colab Notebooks/datasets/train.tar"
!tar xvf "/content/drive/My Drive/Colab Notebooks/datasets/test.tar"
!tar xvf "/content/drive/My Drive/Colab Notebooks/datasets/val.tar"

import os
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import cv2
import numpy as np
import glob

#学習用のラベル0の画像とラベルをそれぞれ配列に格納
imgs1 = []  #画像を入れる配列の初期化
labels1 = []  #ラベルを入れる配列の初期化
files = glob.glob("/content/train/0/*.png")  #train/0/*.pngのファイル名取得
for f in files:  #すべてのファイルに対するfor文
    img = cv2.imread(f)  #画像の読み込み
    imgs1.append(img)  #画像を配列imgs1に追加
i = 1 #iに1を代入
while i < 4487:  #画像枚数文while文で繰り返す
    labels1.append(0)  #ラベル0を配列labels1に追加
    i += 1  #iに1を足す

#学習用のラベル1の画像とラベルをそれぞれの配列に格納
files = glob.glob("/content/train/1/*.png")
for f in files:
    img = cv2.imread(f)
    imgs1.append(img)
i=1
while i < 4495:
    labels1.append(1)
    i += 1
x_train = np.array(imgs1)   
labels = np.array(labels1)
y_train = labels[:, np.newaxis]

#テスト用のラベル0の画像とラベルをそれぞれ配列に格納
imgs2 = []  #画像を入れる配列の初期化
labels2 = []  #ラベルを入れる配列の初期化
files = glob.glob("/content/test/0/*.png")  #test/0/*.pngのファイル名取得
for f in files:  #すべてのファイルに対するfor文
    img = cv2.imread(f)  #画像の読み込み
    imgs2.append(img)  #画像を配列imgs2に追加
i = 1 #iに1を代入
while i < 1196:  #画像枚数文while文で繰り返す
    labels2.append(0)  #ラベル0を配列labels1に追加
    i += 1  #iに1を足す

#テスト用のラベル1の画像とラベルをそれぞれの配列に格納
files = glob.glob("/content/test/1/*.png")
for f in files:
    img = cv2.imread(f)
    imgs2.append(img)
i=1
while i < 1264:
    labels2.append(1)
    i += 1
x_test = np.array(imgs2)   
labels2 = np.array(labels2)
y_test = labels2[:, np.newaxis]

#検証用のラベル0の画像とラベルをそれぞれ配列に格納
imgs3 = []  #画像を入れる配列の初期化
labels3 = []  #ラベルを入れる配列の初期化
files = glob.glob("/content/val/0/*.png")  #val/0/*.pngのファイル名取得
for f in files:  #すべてのファイルに対するfor文
    img = cv2.imread(f)  #画像の読み込み
    imgs3.append(img)  #画像を配列imgs2に追加
i = 1 #iに1を代入
while i < 695:  #画像枚数文while文で繰り返す
    labels3.append(0)  #ラベル0を配列labels1に追加
    i += 1  #iに1を足す

#検証用のラベル1の画像とラベルをそれぞれの配列に格納
files = glob.glob("/content/val/1/*.png")
for f in files:
    img = cv2.imread(f)
    imgs3.append(img)
i=1
while i < 755:
    labels3.append(1)
    i += 1
x_val = np.array(imgs3)   
labels2 = np.array(labels3)
y_val = labels2[:, np.newaxis]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)

# データ型の変換＆正規化
x_train = x_train.astype('float32') / 255  #スケールを揃える
x_test = x_test.astype('float32') / 255
x_val = x_val.astype('float32') / 255
# one-hot変換
num_classes = 2  #クラスの数
y_train = to_categorical(y_train, num_classes = num_classes)  #入力する為の形状に揃える
y_test = to_categorical(y_test, num_classes = num_classes)
y_val = to_categorical(y_val, num_classes = num_classes)

model = Sequential()  #モデルの定義

#畳み込み層
model.add(Conv2D(
    64, # フィルター数（出力される特徴マップのチャネル）
    kernel_size = (4, 4), # フィルターサイズ
    padding = "valid", # 入出力サイズが異なる
    activation = "relu", # 活性化関数
    input_shape = (224, 224, 3) # 入力サイズ
))

#畳み込み層
model.add(Conv2D(
    64,
    kernel_size = (4, 4),
    activation = "relu"
))

#プーリング層 
model.add(MaxPooling2D(pool_size = (4, 4)))  #4×4のフィルタから最大値をとる
model.add(Dropout(0.2))  #過学習を抑える

#畳み込み層
model.add(Conv2D(
    64,
    kernel_size = (4, 4),
    padding = "valid", 
    activation = "relu"
))

#畳み込み層
model.add(Conv2D(
    64,
    kernel_size = 3,
    activation = "relu"
))

#プーリング層
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

# 全結合層（fully-connected layers）につなげるため、
# マトリックスデータ（多次元配列）である特徴マップを多次元ベクトルに変換（平坦化）
model.add(Flatten())
model.add(Dense(512, activation = "relu"))# サイズ512のベクトル（512次元ベクトル）を出力
model.add(Dropout(0.5))
model.add(Dense(num_classes))# クラス数のベクトルを出力
model.add(Activation("softmax"))  #総和1となる確率を算出する関数

optimizer = Adam(lr = 0.001)
model.compile(
    optimizer = optimizer,
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

model.summary()

# EarlyStopping, 監視する値が変化したときに訓練を停止
early_stopping = EarlyStopping( 
    monitor='val_loss',  #監視する値
    patience=10,  #10epochで値が改善されなければ訓練停止
    verbose=1  #ログをプログレスバーで出力
)

# ModelCheckpoint, 各epoch終了時にmodelを保存
#weights_dir = './weights/'
#if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
#model_checkpoint = ModelCheckpoint(
#    weights_dir + "val_loss{val_loss:.3f}.hdf5",
#    monitor = 'val_loss',  #監視する値
#    verbose = 1,
#    save_best_only = False,
#    save_weights_only = False,
#    period = 3  #チェックポイント間の間隔（エポック数）．
#)

# reduce learning rate, 評価値の改善が止まった時に学習率を減らす
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',  #監視する値
    factor = 0.1,  #学習率を減らす割合
    patience = 3,  #3epoch
    verbose = 1  #学習率削減時メッセージを出力
)

# log for TensorBoard
logging = TensorBoard(log_dir = "log/")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # モデルの学習
# hist = model.fit(
#     x_train,
#     y_train,
# 
# 
#     verbose = 1,
#     epochs = 20,
#     batch_size = 32,
#     validation_data = (x_val, y_val),
#     callbacks = [early_stopping, reduce_lr, logging]
# )

model_dir = './model/'
if os.path.exists(model_dir) == False:os.mkdir(model_dir)

model.save(model_dir + 'model.hdf5')

plt.figure(figsize = (18,6))

# accuracy
plt.subplot(1, 2, 1)
plt.plot(hist.history["accuracy"], label = "accuracy", marker = "o")
plt.plot(hist.history["val_accuracy"], label = "val_accuracy", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("accuracy")
#plt.title("")
plt.legend(loc = "best")
plt.grid(color = 'gray', alpha = 0.2)

# loss
plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"], label = "loss", marker = "o")
plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("loss")
#plt.title("")
plt.legend(loc = "best")
plt.grid(color = 'gray', alpha = 0.2)

plt.show()

score = model.evaluate(x_test, y_test, verbose=1)
print("evaluate loss: {0[0]}".format(score))
print("evaluate accuracy: {0[1]}".format(score))

model = load_model(model_dir + 'model.hdf5')

import numpy as np
rounded_labels=np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix
 
predict_classes = model.predict_classes(x_test)
true_classes = rounded_labels
print(confusion_matrix(true_classes, predict_classes))

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
acc_score = accuracy_score(true_classes, predict_classes)
rec_score = recall_score(true_classes, predict_classes, average=None)
pre_score = precision_score(true_classes, predict_classes, average=None)
f1_score = f1_score(true_classes, predict_classes, average=None)
print(acc_score, rec_score, pre_score, f1_score, )

labels = np.array([
   '0','1'
])

# testデータの正解ラベル
true_classes = np.argmax(y_test[0:2458], axis = 1)

# testデータの画像と正解ラベルを出力
fig = plt.figure(figsize = (20, 500))
for i in range(2458):
    plt.subplot(250, 10, i + 1)
    plt.axis('off')
    plt.title(labels[true_classes[i]])
    plt.imshow(x_test[i])
plt.show()

# testデータ30件の予測ラベル
pred_classes = model.predict_classes(x_test[0:2458])

# testデータ30件の予測確率
pred_probs = model.predict(x_test[0:2458]).max(axis = 1)
pred_probs = ['{:.4f}'.format(i) for i in pred_probs]

# testデータ30件の画像と予測ラベル＆予測確率を出力
plt.figure(figsize = (20, 600))
for i in range(2458):
    plt.subplot(250, 10, i + 1)
    plt.axis("off")
    if pred_classes[i] == true_classes[i]:
        plt.title(labels[pred_classes[i]] + '\n' + pred_probs[i])
    else:
        plt.title(labels[pred_classes[i]] + '\n' + pred_probs[i], color = "red")
    plt.imshow(x_test[i])
plt.show()