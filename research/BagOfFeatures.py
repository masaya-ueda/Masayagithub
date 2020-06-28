import cv2
import numpy as np

class BagOfFeatures:
    """This is a class of Bag-of Features by K-means"""
    codebookSize=0
    classifier=None
    def __init__(self, codebookSize):
        self.codebookSize=codebookSize
        self.classifier=cv2.ml.KNearest_create()

    def train(self,features,iterMax=100,term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )):
        retval, bestLabels, codebook=cv2.kmeans(features,self.codebookSize,None,term_crit,10,cv2.KMEANS_RANDOM_CENTERS)
        self.classifier.train(codebook,cv2.ml.ROW_SAMPLE,np.array(range(self.codebookSize)))

    def makeHistgram(self, feature):
        histogram=np.zeros(self.codebookSize)
        dst=np.zeros(self.codebookSize)
        if self.classifier==None :
            raise Exception("You need train this instance.")
        retval, results, neighborResponses, dists=self.classifier.findNearest(feature,1)
        for idx in results:
            idx=int(idx)
            histogram[idx]=histogram[idx]+1
        histogram=cv2.normalize(histogram,dst,norm_type=cv2.NORM_L2)
        #transpose
        histogram=np.reshape(histogram,(1,-1))
        return histogram