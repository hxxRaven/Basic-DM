import requests,re
import json
import time
import csv


url = 'https://service-f9fjwngp-1252021671.bj.apigw.tencentcs.com/release/pneumonia'
html = requests.get(url).text
unicodestr=json.loads(html)  #将string转化为dict
dat = unicodestr["data"].get("statistics")["modifyTime"] #获取data中的内容，取出的内容为str
timeArray = time.localtime(dat/1000)
formatTime = time.strftime("%Y-%m-%d %H:%M", timeArray)


new_list = unicodestr.get("data").get("listByArea")  #获取data中的内容，取出的内容为str

j = 0
print("###############"
      " 数据来源：丁香医生 "
      "###############")
while j < len(new_list):
    a = new_list[j]["cities"]
    s = new_list[j]["provinceName"]

    header = ['时间', '城市', '确诊人数', '疑似病例', '死亡人数', '治愈人数' ]
    with open('./'+s+'.csv', encoding='utf-8-sig', mode='w',newline='') as f:
    #编码utf-8后加-sig可解决csv中文写入乱码问题
        f_csv = csv.writer(f)
        f_csv.writerow(header)
    f.close()

    def save_data(data):
        with open('./'+s+'.csv', encoding='UTF-8', mode='a+',newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(data)
        f.close()

    b = len(a)
    i = 0
    while i<b:
        data = (formatTime)
        confirm = (a[i]['confirmed'])
        city = (a[i]['cityName'])
        suspect = (a[i]['suspected'])
        dead = (a[i]['dead'])
        heal = (a[i]['cured'])

        i+=1
        tap = (data, city, confirm, suspect, dead, heal)
        save_data(tap)

    j += 1
    print(s+"下载结束!")
