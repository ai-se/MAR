from __future__ import print_function, division
import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
import time
import os

class MAR(object):
    def __init__(self):
        self.fea_num = 4000
        self.step = 10
        self.enough = 30


    def create(self,filename):
        self.filename=filename
        self.name=self.filename.split(".")[0]
        self.flag=True
        self.hasLabel=True
        self.record={"x":[],"pos":[]}
        self.body={}
        self.est_num=[]
        self.lastprob=0
        self.offset=0.5
        self.interval=3
        self.buffer=[]

        try:
            ## if model already exists, load it ##
            return self.load()
        except:
            ## otherwise read from file ##
            try:
                self.loadfile()
                self.preprocess()
                self.save()
            except:
                ## cannot find file in workspace ##
                self.flag=False
        return self


    def loadfile(self):
        with open("../workspace/data/" + str(self.filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        header = content[0]
        for field in fields:
            ind = header.index(field)
            self.body[field] = [c[ind] for c in content[1:]]
        try:
            ind = header.index("label")
            self.body["label"] = [c[ind] for c in content[1:]]
        except:
            self.hasLabel=False
            self.body["label"] = ["unknown"] * (len(content) - 1)
        try:
            ind = header.index("code")
            self.body["code"] = np.array([c[ind] for c in content[1:]])
        except:
            self.body["code"]=np.array(['undetermined']*(len(content) - 1))
        return

    def get_numbers(self):
        total = len(self.body["code"])
        pos = Counter(self.body["code"])["yes"]
        neg = Counter(self.body["code"])["no"]
        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if int(pos+neg)>tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
        self.pool = np.where(self.body['code'] == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total

    def export(self):
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "label", "code"]
        with open("../workspace/coded/" + str(self.name) + ".csv", "wb") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(fields)
            for ind in xrange(len(self.body["code"])):
                csvwriter.writerow([self.body[field][ind] for field in fields])
        return

    def preprocess(self):
        ### Combine title and abstract for training ###########
        content = [self.body["Document Title"][index] + " " + self.body["Abstract"][index] for index in
                   xrange(len(self.body["Document Title"]))]
        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore")
        tfidf = tfidfer.fit_transform(content)
        weight = tfidf.sum(axis=0).tolist()[0]
        kept = np.argsort(weight)[-self.fea_num:]
        self.voc = np.array(tfidfer.vocabulary_.keys())[np.argsort(tfidfer.vocabulary_.values())][kept]
        ##############################################################

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
                        vocabulary=self.voc,decode_error="ignore")
        self.csr_mat=tfer.fit_transform(content)
        ########################################################
        return

    ## save model ##
    def save(self):
        with open("memory/"+str(self.name)+".pickle","w") as handle:
            pickle.dump(self,handle)

    ## load model ##
    def load(self):
        with open("memory/" + str(self.name) + ".pickle", "r") as handle:
            tmp = pickle.load(handle)
        return tmp

    def estimate_curve(self,clf):
        ## estimate ##
        # self.est_num=Counter(clf.predict(self.csr_mat[self.pool]))["yes"]
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        order = np.argsort(prob)[::-1]
        tmp = [x for x in np.array(prob)[order] if x > self.offset]

        ind = 0
        sum_tmp = 0
        self.est_num = []
        while True:
            tmp_x = tmp[ind * self.step:(ind + 1) * self.step]
            if len(tmp_x) == 0:
                break
            sum_tmp = sum_tmp + sum(tmp_x) - self.offset * len(tmp_x)
            self.est_num.append(sum_tmp)
            ind = ind + 1
            ##############
        try:
            self.lastprob = np.mean(clf.predict_proba(self.csr_mat[self.buffer])[:,pos_at])
            # self.lastprob = np.mean(np.array(prob)[order][:self.step])
        except:
            pass

    ## Train model ##
    def train(self):
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(self.body['code'] == "yes")[0]
        negs = np.where(self.body['code'] == "no")[0]
        clf.fit(self.csr_mat[self.labeled], self.body['code'][self.labeled])
        ## aggressive undersampling ##
        if len(poses)>=self.enough:

            train_dist = clf.decision_function(self.csr_mat[negs])
            negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
            sample = poses.tolist() + negs[negs_sel].tolist()
            clf.fit(self.csr_mat[sample], self.body['code'][sample])
            self.estimate_curve(clf)

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## Get certain ##
    def certain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1][:self.step]
        return np.array(self.pool)[order],np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        train_dist = clf.decision_function(self.csr_mat[self.pool])
        order = np.argsort(np.abs(train_dist))[:self.step]  ## uncertainty sampling by distance to decision plane
        # order = np.argsort(np.abs(prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)

    ## Format ##
    def format(self,id,prob=[]):
        result=[]
        for ind,i in enumerate(id):
            tmp = {key: self.body[key][i] for key in self.body}
            tmp["id"]=str(i)
            if prob!=[]:
                tmp["prob"]=prob[ind]
            result.append(tmp)
        return result

    ## Code candidate studies ##
    def code(self,id,label):
        self.buffer.append(id)
        self.buffer=self.buffer[-self.step * self.interval:]
        self.body["code"][id]=label

    ## Plot ##
    def plot(self):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        fig = plt.figure()
        plt.plot(self.record['x'], self.record["pos"])
        ### estimation ####
        if len(self.est_num)>0 and self.lastprob>self.offset:
            der = (self.record["pos"][-1]-self.record["pos"][-1-self.interval])/(self.record["x"][-1]-self.record["x"][-1-self.interval])
            xx=np.array(range(len(self.est_num)+1))
            yy=map(int,np.array(self.est_num)*der/(self.lastprob-self.offset)+self.record["pos"][-1])
            # yy = map(int, np.array(self.est_num) + (der - self.lastprob)*xx[1:]*self.step + self.record["pos"][-1])
            yy=[self.record["pos"][-1]]+list(yy)
            xx=xx*self.step+self.record["x"][-1]
            plt.plot(xx, yy, "-.")
        ####################
        plt.ylabel("Relevant Found")
        plt.xlabel("Documents Reviewed")
        name=self.name+ "_" + str(int(time.time()))+".png"
        plt.savefig("./static/image/" + name)
        plt.close(fig)
        return name



