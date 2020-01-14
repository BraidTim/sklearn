# -*- coding: utf-8 -*-
import MySQLdb as mdb

con = mdb.connect('localhost', 'root',
'2O12o117', 'tblive',charset='utf8');
#所有的查询，都在连接 con 的一个模块 cursor 上面运行的
cur = con.cursor()
#执行一个查询
cur.execute("SELECT * from goods_anchor_live limit 1")
#取得上个查询的结果，是单个结果
data = cur.fetchall()
print ("Database version : %s " % data)

con.close()