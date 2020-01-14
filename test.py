

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import datasets, linear_model,tree

train_test = 5
knn_N = 5
alg = 2
import psycopg2
x_list=["avgcomment","avglike","avgjoincount","avgmaxonline","fanscount","beginbuycount","commentcount",
        "favcount","likecount","livecount","price","total30salecount"]
x_usefull_list = ["avglike","avgjoincount","avgmaxonline","beginbuycount","commentcount","price","total30salecount"]
x_usefull_list = ["avgmaxonline"]

y_list = ["cls"]
conn = psycopg2.connect(database="ERP_Database", user="BraidTim", password="2O12o117", host="127.0.0.1", port="5432")
cur = conn.cursor()

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
        if alg == 1:
            regr = linear_model.LinearRegression(True, True, True)
        if alg == 2:
            regr = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=20 )

            #regr = tree.DecisionTreeRegressor(max_depth=5, min_samples_leaf=20 )

        # regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X_train, Y_train)

        # The coefficients
        if alg==1:
            print('Coefficients: \n', regr.coef_)
            coef[i] = regr.coef_[0]
        # The mean square error
        temp = np.mean((regr.predict(X_test) - Y_test) ** 2)
        if regr.predict(X_test) == Y_test:
            bingo+=1
        else:
            miss+=1
        print("Residual sum of squares: %.2f"
              % temp)
        total_var[i] = temp
        # Explained variance score: 1 is perfect prediction
        if alg == 1:
            print('Variance score: %.2f' % regr.score(X_test, Y_test))
    print("test:")
    print(bingo)
    print(miss)
    print("final Residual sum of squares: "+str(total_var))

    print("final avg Residual sum of squares: "+str(total_var.sum(axis=0)/train_test))
    print("final coef :")
    print(coef)
    print(coef.sum(axis=0) / train_test)
x = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
re = depart_data(x,3)
def get_x(col_name):

    sql = "select cast("+col_name+" as float) from hth_anchor_to_goods_index_class_random_1w"
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

# X_T = []
# for elem in x_list:
#
#     X_T.append(get_x(elem).reshape(-1, 1))
#     # X_T.append(get_x('avgjoincount').reshape(-1, 1))
#     # X_T.append(get_x('avgmaxonline').reshape(-1, 1))
#     # X_T.append(get_x('fanscount').reshape(-1, 1))
#     # X_T.append(get_x('score').reshape(-1, 1))
# X = X_T[0]
# for i in range(1,len(X_T)):
#     X = np.hstack((X,X_T[i]))
#



#----------------coef--------------
if 0 :
    for elem in x_list:
        X_T = []
        X_T.append(get_x(elem).reshape(-1, 1))
        # X_T.append(get_x('avgjoincount').reshape(-1, 1))
        # X_T.append(get_x('avgmaxonline').reshape(-1, 1))
        # X_T.append(get_x('fanscount').reshape(-1, 1))
        # X_T.append(get_x('score').reshape(-1, 1))
        X = X_T[0]
        for i in range(1,len(X_T)):
            X = np.hstack((X,X_T[i]))


        a = np.sum((X - X.mean()) * (Y - Y.mean()))
        b = math.sqrt(np.sum( (X-X.mean())*(X-X.mean()) ))
        c = math.sqrt(np.sum( (Y-Y.mean())*(Y-Y.mean()) ))
        print(elem+":" +str(a / b / c))
#-----------------------------------------
# print(a)
# print(b)
# print( c )
# print (a/b/c)



if 0:
    # X = np.array([1,2,3,4,5,6,7,8,9])
    # Y = np.array([2,4,6,7,10,12,14,16,19])
    len = len(X)
    split = math.floor(len/train_test)

    #print(diabetes_X)
    # Split the data into training/testing sets
    X_train = X[:-1*split]
    X_test = X[-1*split:]


    # Split the targets into training/testing sets
    Y_train = Y[:-split].reshape(-1, 1)
    Y_test = Y[-split:].reshape(-1, 1)

    # Create linear regression object
    regr = linear_model.LinearRegression(True,True,True)

    #regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((regr.predict(X_test) - Y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X_test, Y_test))

    if 1:
    # Plot outputs
        plt.scatter(X_test, Y_test,  color='black')
        plt.plot(X_test, regr.predict(X_test), color='blue',
                 linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.show()

