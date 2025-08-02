import pandas as pd
import numpy as np
import math

def get_data(path):
    return pd.read_csv(path)

df = get_data('watermalonBook\decisionTree\西瓜数据集3.0.txt')

D = df.iloc[:,1:]


class TreeNode():
    def __init__(self,label = None,branch = {},condition = None):
        self.label = label
        self.branch = branch
        self.condition = condition
    
def NodeLabel(label_arr):
    label_count = {}       # store count of label 
      
    for label in label_arr:
        if label in label_count: label_count[label] += 1
        else: label_count[label] = 1
        
    return label_count

class DecisionTree():
    def __init__(self):
        self.root = None
    def forward(self):
        pass
    def Ent(self,D):
        n_D = len(D)
        label_arr = D.iloc[:,-1]
        label_count = NodeLabel(label_arr)
        ans = 0
        for _,value in label_count.items():
            p_k = value/n_D
            ans += p_k*math.log2(p_k)
        return -ans
    
    def Gain(self,D,a):
        if D[a].dtype in (int,float):
            D = D.sort_values(a)
            D = D.reset_index()
            t_a = []
            for i in range(len(D[a])-1):
                t_a.append((D[a][i]+D[a][i+1])/2)
            res = 0
            best_t = 0
            for t in t_a:
                D_v_p = D[D[a]>t]
                D_v_n = D[D[a]<=t]
                gain = self.Ent(D)-self.Ent(D_v_p)*len(D_v_p)/len(D) - self.Ent(D_v_n)*len(D_v_n)/len(D)
                if gain>res:
                    res = gain
                    best_t = t
            return res,best_t
        else:
            res = 0
            features = D[a]
            for k in set(features):
                D_v = D[D[a]==k]
                res -= self.Ent(D_v)*len(D_v)/len(D)
            res+=self.Ent(D)
            return res,None
        
    def Gini(self,D):
        n_D = len(D)
        label_arr = D.iloc[:,-1]
        label_count = NodeLabel(label_arr)
        ans = 0
        for _,value in label_count.items():
            p_k = value/n_D
            ans += p_k*p_k
        return 1-ans
    
    def Gini_index(self,D,a):
        '''
        计算Gini_index
        '''
        if D[a].dtype in (int,float):
            D = D.sort_values(a)
            D = D.reset_index()
            t_a = []
            for i in range(len(D[a])-1):
                t_a.append((D[a][i]+D[a][i+1])/2)
            res = 0
            best_t = 0
            for t in t_a:
                D_v_p = D[D[a]>t]
                D_v_n = D[D[a]<=t]
                gini = self.Gini(D_v_p)*len(D_v_p)/len(D) + self.Gini(D_v_n)*len(D_v_n)/len(D)
                if gini<res:
                    res = gini
                    best_t = t
            return res,best_t
        else:
            res = 0
            features = D[a]
            for k in set(features):
                D_v = D[D[a]==k]
                res += self.Gini(D_v)*len(D_v)/len(D)
            return res,None

    def optimArr(self,D,A,type = 'GainEnt'):
        if type == 'GainEnt':
            max_a = None
            max_gain = 0.0
            best_t = None
            for a in A:
                curGain,t = self.Gain(D,a)
                if curGain>max_gain:
                    max_gain = curGain
                    max_a = a
                    best_t = t
            return max_a,best_t
        elif type == 'gini':
            min_a = None
            min_gini = 1e9
            best_t = None
            for a in A:
                curGini,t = self.Gini_index(D,a)
                if curGini<min_gini:
                    min_gini = curGini
                    min_a = a
                    best_t = t
            return min_a,best_t
        elif type == 'GainRatio':
            pass

    def generateTree(self,D):
        '''
        D = {(x_i,y_i)...}训练集
        A = {a_i}属性集
        '''
        label_arr = D.iloc[:,-1]
        label_count = NodeLabel(label_arr)
        A = D.columns[:-1]
        new_node = TreeNode(None,{},None)
        if label_count:
            if len(label_count)==1:
                new_node.label = max(label_count,key=label_count.get)
                return new_node
            if len(label_arr)==0:
                new_node.label = max(label_count,key=label_count.get)
                return new_node
            # 划分最优属性
            a_star,t = self.optimArr(D,A,type='gini')
            new_node.condition = (a_star,t)
            if t==None:
                features = set(D[a_star])
                for k in features:
                    D_v = D[D[a_star]==k]
                    D_v = D_v.drop(a_star,axis = 1)
                    new_node.branch[k] = self.generateTree(D_v)
            elif t is not None:
                D_v_l = D[D[a_star]<=t].drop(a_star,axis = 1)
                D_v_r = D[D[a_star]>t].drop(a_star,axis = 1)
                v_l = '<=%.3f'%t
                v_r = '>%.3f'%t
                new_node.branch[v_l] = self.generateTree(D_v_l)
                new_node.branch[v_r] = self.generateTree(D_v_r)
        self.root = new_node
        return new_node
    def predict(self,x):
        root = self.root
        while root.condition is not None:
            if root.condition[-1] != None:
                c,t = root.condition
                if x[c]>t:
                    key = '>%.3f'%t
                    root = root.branch[key]
                else:
                    key = '<=%.3f'%t
                    root = root.branch[key]
            else:
                c,_ = root.condition
                key = x[c]
                root = root.branch[key]
        return root.label
