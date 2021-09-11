from pymongo import MongoClient
from time import time
import numpy as np
from math import log10
from scipy import sparse as sp
import matplotlib.pyplot as plt
import pickle


def SybilRank(day):
    
    client = MongoClient('localhost', 27017)
    db = client['tweetsdb']
    minimum_retweet_number = 3
    red_dirigida = [
    {"$match": {"$expr":{"$ne":["$retweeted_status", None]}}},
    {"$match": {"$expr":{"$ne":["$retweeted_status.user.screen_name", 
                                "$user.screen_name"]}}},
    {"$addFields": { "ts": { "$toDate": 
                            { "$multiply": ['$created_at', 1000] } } }},
    {"$addFields":{"month":{"$dateToString":{"date": '$ts',  "format": '%m', 
                                             "timezone": 'Europe/Madrid' }}}},
    {"$addFields":{"day":{"$dateToString":{"date": '$ts',  "format": '%d', 
                                           "timezone": 'Europe/Madrid' }}}},
    {"$match": {"$expr":{"$eq":["$month", "04"]}}},
    {"$match": {"$expr":{"$gte":["$day", "12"]}}},
    {"$match": {"$expr":{"$lte":["$day", "20"]}}},
    {"$group": {
            "_id": "$user.screen_name",
            "friends": { "$addToSet":"$retweeted_status.user.screen_name"},
            "retweets":{"$sum":1},
            "verified":{"$first":"$user.verified"}}},
    {"$match": {"$expr":{"$gte":["$retweets", minimum_retweet_number]}}},
    ]
    res = list(db.tweets.aggregate(red_dirigida, allowDiskUse = True))
    
    directed_network = []
    pos_dict = {}; deg_dict = {} 
    retweets_dict = {}; verified_dict = {}
    nodesList = []; friendsList = []
    verified_count = 0
    edgescount = 0
    for i in range(0, len(res)):
        node = res[i]["_id"]; friends = res[i]["friends"]
        directed_network.append([node, friends])
        nodesList.append(node); friendsList.append(friends)
        pos_dict.update({node:i})
        deg_dict.update({node:len(friends)}); edgescount += len(friends)
        retweets_dict.update({node:res[i]["retweets"]})
        if res[i]["verified"] != True:
            verified_dict.update({node:False})
        else:
            verified_dict.update({node:True})
            verified_count += 1
    
    c = len(nodesList)
    for i in range(0, len(directed_network)):
        for j in range(0, len(directed_network[i][1])):
            sink = directed_network[i][1][j]
            try:
                aux = pos_dict[sink]
            except KeyError:
                nodesList.append(sink)
                friendsList.append([])
                pos_dict.update({sink:c}); c += 1
                deg_dict.update({sink:0})
                retweets_dict.update({sink:0})
                verified_dict.update({sink:False})
    print("There are", len(nodesList), "nodes in our network and",
          edgescount, "edges")
    r1 = len(nodesList); r2 = edgescount
    
    adjacencyMatrix = sp.coo_matrix((len(nodesList), len(nodesList)))
    adjacencyMatrix = sp.lil_matrix(adjacencyMatrix)

    trustVector = sp.coo_matrix((1, len(nodesList)))
    trustVector = sp.lil_matrix(trustVector)

    for i in range(0, len(nodesList)):
        degree = deg_dict[nodesList[i]]
        if degree == 0:
            adjacencyMatrix[i,i] = 1 #Ponemos un 1 para el selfloop
        else:
            for j in range(0, degree):
                pos = pos_dict[friendsList[i][j]]
                adjacencyMatrix[i,pos] = 1/degree

        trustable_seed = verified_dict[nodesList[i]]
        if trustable_seed == True:
            trustVector[0,i] = 1
        else:
            trustVector[0,i] = 0

    adjacencyMatrix = sp.csr_matrix(adjacencyMatrix)
    trustVector = sp.csr_matrix(trustVector)
    
    print("Initial trust is", np.sum(trustVector.toarray()))
    niter = int(np.round(log10(len(nodesList))))
    for i in range(0, niter):
        trustVector = trustVector.dot(adjacencyMatrix)
    print("Final trust is", np.sum(trustVector.toarray())) 
        
    trustVector = trustVector.toarray().tolist()[0]
    for i in range(0, len(trustVector)):
        degree = deg_dict[nodesList[i]]
        if degree != 0:
            trustVector[i] /= degree

    zipped_lists = zip(trustVector, nodesList)
    sorted_pairs = sorted(zipped_lists, key = lambda x: x[0], reverse = True)

    trustVector = {}
    for tup in sorted_pairs:
        trustVector.update({tup[1]:tup[0]})
        
    print(evaluate(trustVector, verified_dict))
    
    
class SybilRankTemporal:
    def __init__(self, database = "mini", day = "20"):
        self.params = ""
        if database != "load":
            client = MongoClient('localhost', 27017)
            db = client['tweetsdb']

            pipeline = [
            {"$match": {"$expr":{"$ne":["$retweeted_status", None]}}},
            {"$addFields": { "ts": { "$toDate": { "$multiply":
                                                 ['$created_at', 1000] } } }},
            {"$addFields": {"month": {"$dateToString": {"date": 
                                                        '$ts',  "format": '%m',  "timezone": 'Europe/Madrid' }}}},
            {"$addFields": {"day": {"$dateToString": {"date": 
   '$ts',  "format": '%d',  "timezone": 'Europe/Madrid' }}}},
            {"$match": {"$expr":{"$eq":["$month", "04"]}}},
            {"$match": {"$expr":{"$eq":["$day", day]}}},
            {"$project":{"ts":1, "_id":0, "user.screen_name":1,
                         "user.verified":1,
                         "retweeted_status.user.screen_name":1,
                         "retweeted_status.user.verified":1,
                         "user.followers_count":1,
                         "retweeted_status.user.followers_count":1,
                         "user.friends_count":1,
                         "retweeted_status.user.friends_count":1
                         }}]

            if database == "mini":
                objects = list(db.mini.aggregate(pipeline))
            elif database == "sample":
                objects = list(db.tweets_sample.aggregate(pipeline))
            else:
                objects = list(db.tweets.aggregate(pipeline))

            data = []
            for x in objects:
                try:
                    a = x["user"]["verified"]
                except KeyError:
                    x["user"]["verified"] = False
                try:
                    a = x["retweeted_status"]["user"]["verified"]
                except KeyError:
                    x["retweeted_status"]["user"]["verified"] = False  
                data.append([x["user"]["screen_name"],
                             x["user"]["verified"],
                             x["retweeted_status"]["user"]["screen_name"],
                             x["retweeted_status"]["user"]["verified"],
                             x["ts"].hour*60 + x["ts"].minute + 
                             x["ts"].second/60,
                             x["user"]["followers_count"],
                             x["retweeted_status"]["user"]["followers_count"],
                             x["user"]["friends_count"],
                             x["retweeted_status"]["user"]["friends_count"]
                             ])

            self.data = sorted(data, key = lambda x: x[4], reverse=False)
            self.verified = {}
            self.followers = {}
            for tweet in self.data:
                self.verified.update({tweet[0]:tweet[1]})
                self.verified.update({tweet[2]:tweet[3]})
                self.followers.update({tweet[0]:tweet[5]})
                self.followers.update({tweet[2]:tweet[6]})
        else:
            self.data = pickle.load(open(day + "data.pkl", "rb"))
            self.verified = pickle.load(open(day + "verified.pkl", "rb"))
            self.followers = pickle.load(open(day + "followers.pkl", "rb"))

    def saveData(self, day):
        with open(day + "data.pkl", "wb") as f:
            pickle.dump(self.data, f)
        with open(day + "verified.pkl", "wb") as f:
            pickle.dump(self.verified, f)
        with open(day + "followers.pkl", "wb") as f:
            pickle.dump(self.followers, f)
            

    def getWindowData(self, windowDuration = 60):
        self.params += f"windowDuration={windowDuration}\n"
        #Creamos una lista de listas, cada lista es un intervalo temporal
        #que contiene todos los retweets que se han dado dentro de él
        #WindowDuration está en minutos
        intervals = 24*60/windowDuration
        self.wData = [[] for i in range(int(intervals))]
        i = 0; current = windowDuration
        for x in self.data:
            if x[4]>current:
                current += windowDuration
                i += 1
                self.wData[i].append(x[0:4])
            else:
                self.wData[i].append(x[0:4])

    def saveTrustVector(self, day):
        with open(day + "trust.pkl", "wb") as f:
            pickle.dump(self.trust)

    def getTrustVector(self):
        return self.trust, self.verified, self.followers

    def loadTrustVector(self, day):
        self.trust = pickle.load(open(day + "trust.pkl", "rb"))

    def setStandardTrust(self):
        self.params += f"trust=standard\n"
        self.trust = {}
        for window in self.wData:
            for tweet in window:
                #Diccionario de nodos y confianza inicial
                value = 1 if tweet[1] == True else 0
                try:
                    x = self.trust[tweet[0]]
                except KeyError:
                    self.trust.update({tweet[0]:value})

                value = 1 if tweet[3] == True else 0
                try:
                    x = self.trust[tweet[2]]
                except KeyError:
                    self.trust.update({tweet[2]:value})

    def setConstantTrust(self, prop = 100, initial = 10):
        self.params += f"trust=constant\n"
        self.trust = {}
        for tweet in self.data:
            #Diccionario de nodos y confianza inicial
            try:
                x = self.trust[tweet[0]]
            except KeyError:
                self.trust.update({tweet[0]:initial+tweet[5]/prop})

            try:
                x = self.trust[tweet[2]]
            except KeyError:
                self.trust.update({tweet[2]:initial+tweet[6]/prop})
                
    def setSameTrust(self, value = 1):
        self.params += f"trust=same\n"
        self.trust = {}
        for tweet in self.data:
            #Diccionario de nodos y confianza inicial
            try:
                x = self.trust[tweet[0]]
            except KeyError:
                self.trust.update({tweet[0]:value})

            try:
                x = self.trust[tweet[2]]
            except KeyError:
                self.trust.update({tweet[2]:value})

    def setRatioTrust(self):
        self.params += f"trust=ratio\n"
        self.trust = {}
        for tweet in self.data:
            #Diccionario de nodos y confianza inicial
            try:
                x = self.trust[tweet[0]]
            except KeyError:
                try:
                    self.trust.update({tweet[0]:tweet[5]/tweet[7]})
                except ZeroDivisionError:
                    self.trust.update({tweet[0]:tweet[5]/1})

            try:
                x = self.trust[tweet[2]]
            except KeyError:
                try:
                    self.trust.update({tweet[2]:tweet[6]/tweet[8]})
                except ZeroDivisionError:
                    self.trust.update({tweet[2]:tweet[6]/1})
                    
    def setFollowersTrust(self):
        self.params += f"trust=followers\n"
        self.trust = {}
        for tweet in self.data:
            #Diccionario de nodos y confianza inicial
            try:
                x = self.trust[tweet[0]]
            except KeyError:
                self.trust.update({tweet[0]:tweet[5]})

            try:
                x = self.trust[tweet[2]]
            except KeyError:
                self.trust.update({tweet[2]:tweet[6]})

    def totalTrust(self):
        total = 0
        for x in self.trust.values():
            total += x
        print(total)

    def temporalSybilRank(self, punishment, proportion, unity):
        self.params += f"punishment={punishment}\nproportion={proportion}       \nunity={unity}\n"
        for window in self.wData:
            dim = int(len([item for sublist in window for item in sublist])/2)

            am = sp.coo_matrix((dim, dim))
            am = sp.lil_matrix(am)

            nodes = {}; reverseNodes = {}; i = 0; windowTrust = []
            for tweet in window:
                #Diccionario de nodos y posiciones
                try:
                    x = nodes[tweet[0]]
                except KeyError:
                    nodes.update({tweet[0]:i}); i += 1
                    reverseNodes.update({i-1:tweet[0]})

                try:
                    x = nodes[tweet[2]]
                except KeyError:
                    nodes.update({tweet[2]:i}); i += 1
                    reverseNodes.update({i-1:tweet[2]})

                am[nodes[tweet[0]], nodes[tweet[2]]] += 1
                if unity == False:
                #O bien se transmite la confianza entera o un parte de ella
                    trust1 = self.trust[tweet[0]]/proportion
                    trust2 = self.trust[tweet[2]]/proportion
                else:
                #O bien confianza unidad para transmitir o bien nula
                    trust1 = 1 if self.trust[tweet[0]] != 0 else 0
                    trust2 = 1 if self.trust[tweet[2]] != 0 else 0
                windowTrust.append(trust1)
                windowTrust.append(trust2)

            for i in range(0, am.shape[0]):
                am[i,i] = 1

            am = sp.csr_matrix(am); windowTrust = sp.csr_matrix(windowTrust)

            #Calculo la diferencia de confianzas antes y después de SybilRank
            windowDifference = windowTrust.dot(am)-windowTrust

            #La idea es ahora sumar esa diferencia al vector de confianzas general
            for i in range(len(nodes)):
                self.trust[reverseNodes[i]] += windowDifference[0,i]

            #En cada ventana se "castiga" a todos los usuarios
            #De modo que si no reciben confianza, seguirán perdiendo
            if punishment != 1:
                for key in self.trust.keys():
                    self.trust[key] *= punishment

    def getTop(self):
        SR = dict(sorted(self.trust.items(), key=lambda item: item[1], reverse = True))
        i = 0
        for key in SR.keys():
            print(key, SR[key])
            i += 1
            if i == 10:
                break

    def getBottom(self):
        SR = dict(sorted(self.trust.items(), key=lambda item: item[1], reverse = False))
        i = 0
        for key in SR.keys():
            print(key, SR[key])
            i += 1
            if i == 10:
                break
            
    def updateExistingTrust(self, trust):
        for key, value in trust.items():
            try:
                self.trust[key] = value
            except KeyError:
                pass
    
def evaluate(trust, verifiedVector, botometerThreshold = .95):
    client = MongoClient('localhost', 27017)
    db = client['Botometer']

    pipeline = [{"$group": {"_id":"$name","cap":{"$first":"$cap.universal"}}}]
    data = list(db.user.aggregate(pipeline, allowDiskUse = True))

    actual = {}
    for x in data:
        if x["cap"] != None:
            try:
                verified = verifiedVector[x["_id"]]
            except KeyError:
                verified = False
            if x["cap"] > botometerThreshold and verified == False:
                actual.update({x["_id"]:True})
            else:
                actual.update({x["_id"]:False})
    SR = dict(sorted(trust.items(), key=lambda item: item[1], reverse =False))

    predicted = {}
    thresholds = np.linspace(0, 1, 21)
    tpr = []; fpr = []
    for threshold in thresholds:
        fp = 0; tp = 0; fn = 0; tn = 0
        for i, key in enumerate(SR.keys()):
            if i <= int(threshold*len(SR)):
                predicted.update({key:True})
            else:
                predicted.update({key:False})
                if verifiedVector[key] == True:
                    tn += 1

        for key in actual.keys():
            try: 
                if actual[key] == True and predicted[key] == True:
                    tp += 1
                elif actual[key] == False and predicted[key] == True:
                    fp += 1
                elif actual[key] == False and predicted[key] == False:
                    tn += 1
                else:
                    fn += 1
            except KeyError:
                pass

        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
    auc = round(np.trapz(tpr, fpr),3) 
    plt.plot(fpr, tpr, '-k')
    plt.text(0.7, 0.2, str(auc), fontsize=16)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
    plt.show()
    return auc