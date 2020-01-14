x=[]
y=[]
import re
import psycopg2
x_list=["avgcomment","avglike","avgjoincount","avgmaxonline","fanscount","beginbuycount","commentcount",
        "favcount","likecount","livecount","price","total30salecount"]
x_usefull_list = ["index","avglike","avgjoincount","avgmaxonline","beginbuycount","commentcount","price","total30salecount"]
y_list = ["salecount"]
conn = psycopg2.connect(database="ERP_Database", user="BraidTim", password="2O12o117", host="127.0.0.1", port="5432")
cur = conn.cursor()
def isnumber(num):

    regex = re.compile(r"^(-?\d+)(\.\d*)?$")

    if re.match(regex,num):

        return True

    else:

        return False
print("input x:")
while 1:
    p = input()
    if isnumber(p) == False:
        print(p)
        break
    else:
        x.append(float(p))

print("input y:")

while 1:
    p = input()
    if isnumber(p) == False:
        print(p)
        break
    else:
        y.append(float(p))
#linear
#str="<!DOCTYPE html><html><head> <meta charset='utf-8'> <title>ECharts</title>  <script src='echarts.js'></script></head><body> <div id='main' style='width: 1000px;height:1000px;'></div> <script type='text/javascript'> var myChart = echarts.init(document.getElementById('main')); option = { xAxis: { type: 'category', data: "+str(x)+" }, yAxis: { type: 'value' }, series: [{ data: "+str(y)+", type: 'line' }]};console.log(option); myChart.setOption(option); </script></body></html>"
#scatter

xy = []
x_y = ['index','log']

if len(x) == 0:
    sql = "select cast(" + x_y[0] + " as float),cast(" + x_y[1] + " as float) from temp_anchor_e_b" \
                                                                  ""
    cur.execute(sql)
    res = cur.fetchall()
    for i in range(len(res)):
        xy.append([res[i][0],res[i][1]])
    pass

else:
    for i in range(len(x)):
        xy.append([x[i],y[i]])
str="<!DOCTYPE html><html><head> <meta charset='utf-8'> <title>ECharts</title><script src='echarts.js'></script></head><body> <div id='main' style='width: 1000px;height:1000px;'></div> <script type='text/javascript'> var myChart = echarts.init(document.getElementById('main')); option = { xAxis: {}, yAxis: {}, series: [{ symbolSize: 5, data: "+str(xy)+", type: 'scatter' }]}; myChart.setOption(option); </script></body></html>"
file = open('C:/work/nginx-1.14.1/html/scatter4.html','w')
file.writelines(str)
file.close()