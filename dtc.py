

import matplotlib.pyplot as plt
import numpy as np
import math
import os
from sklearn import datasets, linear_model,tree

train_test = 5
knn_N = 5
alg = 2
import psycopg2
import MySQLdb as mdb
x_list=["avgcomment","avglike","avgjoincount","avgmaxonline","fanscount","beginbuycount","commentcount",
        "favcount","likecount","livecount","price","total30salecount"]
x_usefull_list = ["index","avglike","avgjoincount","avgmaxonline","beginbuycount","commentcount","price","total30salecount","viewcount"]
x_usefull_list = ["row_index","avglike","avgjoincount","avgmaxonline","beginbuycount","price","viewcount","maxonline","live_salecount"]

#x_usefull_list = ["avgmaxonline"]

y_list = ["cls"]
#conn = psycopg2.connect(database="ERP_Database", user="BraidTim", password="2O12o117", host="127.0.0.1", port="5432")
conn = mdb.connect('localhost', 'root','2O12o117', 'tblive');
cur = conn.cursor()
table_name = 'goods_anchor_live_temp_1k_index'
def depart_data(M,parts):
    R = []
    for i in range(parts):
        R.append(np.array([]))
    for i in range(len(M)):
        if len(R[i%parts]) == 0:
            R[i % parts] = M[i]
        else:
            R[i % parts] = np.vstack((R[i%parts],M[i]))
    return R

def loop_linear(Rx,Ry):
    total_var = np.zeros((train_test,1),dtype=float)
    coef = np.zeros((train_test,len(x_usefull_list)),dtype=float)
    bingo = 0
    miss = 0
    predict_fre = [0,0,0,0]
    for i in range(len(Rx)):
        X_train = np.array([])
        Y_train = np.array([])

        X_test = Rx[i]
        Y_test = Ry[i]
        for j in range(len(Rx)):
            if i != j:
                if len(X_train)==0:
                    X_train = Rx[j]
                    Y_train = Ry[j]
                else:
                    X_train = np.vstack((X_train,Rx[j]))
                    Y_train = np.vstack((Y_train,Ry[j]))
       # Create linear regression object

        regr = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=20 )

            #regr = tree.DecisionTreeRegressor(max_depth=5, min_samples_leaf=20 )

        # regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X_train[:,1:], Y_train)
        file_str = "dtc"+str(i)+".dot"
        with open(file_str,'w') as f:
            tree.export_graphviz(regr,out_file=f)
        os.system(r"cd C:\projects\python\sklearn")
        cmd = r"C:\projects\python\sklearn\graphviz-2.38\release\bin\dot -Tpdf "+file_str+" -o "+str(i)+".pdf"
        os.system(cmd)
        for ii in range(len(X_test)):
            pre = regr.predict([X_test[ii][1:]])
            sql = "update "+table_name+" set DT_predict = " + str(pre[0]) + " where row_index = " + str(X_test[ii][0])
            cur.execute(sql)
            pass


x = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
re = depart_data(x,3)
def get_x(col_name):

    sql = "select cast("+col_name+" as signed) from "+table_name+""
    cur.execute(sql)
    res = cur.fetchall()
    x = []
    for elem in res:
        x.append(elem[0])
    return np.array(x).reshape(-1,1)
# Load the diabetes dataset
Y = get_x('salecount').reshape(-1, 1)

def linear_reg_with_colname_list(col_name_list_x,col_name_list_y,method):
    X_data = np.array([])
    Y_data = np.array([])

    for elem in col_name_list_x:
        if len(X_data) == 0:
            X_data = get_x(elem)
        else:
            X_data = np.hstack((X_data,get_x(elem)))

    for elem in col_name_list_y:
        if len(Y_data) == 0:
            Y_data = get_x(elem)
        else:
            Y_data = np.hstack((Y_data,get_x(elem)))

    Rx = depart_data(X_data,train_test)
    Ry = depart_data(Y_data,train_test)
    if method == 0:
        loop_linear(Rx,Ry)



linear_reg_with_colname_list(x_usefull_list,y_list,0)
conn.commit()
