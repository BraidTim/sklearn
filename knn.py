

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import datasets, linear_model
from sklearn.preprocessing import normalize

train_test = 5
knn_N = 10
import psycopg2
x_list=["avgcomment","avglike","avgjoincount","avgmaxonline","fanscount","beginbuycount","commentcount",
        "favcount","likecount","livecount","price","total30salecount"]
x_usefull_list = ["index","avglike","avgjoincount","avgmaxonline","beginbuycount","commentcount","price","total30salecount"]
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

def get_x(col_name):

    sql = "select cast("+col_name+" as float) from hth_anchor_to_goods_index_class_random_1w_knn"
    cur.execute(sql)
    res = cur.fetchall()
    x = []
    for elem in res:
        x.append(elem[0])
    return np.array(x).reshape(-1,1)
# Load the diabetes dataset
Y = get_x('cls').reshape(-1, 1)

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
        pass
    if method == 1:
        knn(Rx,Ry)

def knn_predict(X_train,Y_train,X):
    #print("X : " + str(X.reshape(1,-1)))
    X_diff = X_train[:,1:] - X[1:]
    X_norm = normalize(abs(X_diff),axis=0,norm='max')
    #print("X_diff : " + str(X_diff))
    X_dis = np.linalg.norm(X_norm, axis=1)
    #print("X_dis : " + str(X_dis))

    top_N_index = np.argpartition(X_dis, knn_N)[:knn_N]
    sql = "update hth_anchor_to_goods_index_class_random_1w_knn set knn_list = '"+str(top_N_index) +"' where index = " + str(X[0])
    #cur.execute(sql)
    #print("top "+str(knn_N)+" X : "+str(X_train[top_N_index].reshape(5,-1)))

    #print("top "+str(knn_N)+" Y : "+str(Y_train[top_N_index].reshape(1,-1)))
    predicted = np.sum(Y_train[top_N_index])/knn_N
    predicted = np.argmax(np.bincount(Y_train[top_N_index].reshape(1,-1).tolist()[0]))

    #print("predict : "+str(predicted))
    sql = "update hth_anchor_to_goods_index_class_random_1w_knn set knn_predict = " + str(predicted)+ "where index = " + str(X[0])
    cur.execute(sql)
    return predicted

def avg_predict(X_train,Y_train,X):
    return np.mean(Y_train)

# XX = np.array([[1,2],[3,4],[5,6],[7,8]])
# YY = np.array([1,2,3,4]).reshape(-1,1)
# Z = knn_predict(XX,YY,[2,3])
def knn(Rx,Ry):
    total_RSS = np.zeros((train_test, 1), dtype=float)
    #every time use ith as test and others as train
    for i in range(len(Rx)):
        # if i != 4:
        #     continue
        X_train = np.array([])
        Y_train = np.array([])

        X_test = Rx[i]
        Y_test = Ry[i]
        for j in range(len(Rx)):
            if i != j:
                if len(X_train) == 0:
                    X_train = Rx[j]
                    Y_train = Ry[j]
                else:
                    X_train = np.vstack((X_train, Rx[j]))
                    Y_train = np.vstack((Y_train, Ry[j]))


        sample_R_S_S = np.zeros((len(X_test),1))
        for j in range(len(X_test)):
            # if X_test[j][0]!=10000:
            #     continue
            y_pre = knn_predict(X_train,Y_train,X_test[j])
            #y_pre = avg_predict(X_train,Y_train,X_test[j])

            #print("real : "+str(Y_test[j]))
            sample_R_S_S[j] = (y_pre - Y_test[j])**2
        total_RSS[i] = np.mean(sample_R_S_S)

    print("final Residual sum of squares: " + str(total_RSS))

    print("final avg Residual sum of squares: " + str(np.sum(total_RSS) / train_test))


    pass

linear_reg_with_colname_list(x_usefull_list,y_list,1)
conn.commit()

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

