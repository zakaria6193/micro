from flask import Flask, request,jsonify, session
import datetime as datedatedate
from datetime import datetime
from datetime import date
from xgboost import XGBRegressor
import xgboost




import requests
from bs4 import BeautifulSoup

from flask_cors import CORS
import math


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os





import dateutil


import pyrebase
import numpy as np
from flask import session

app = Flask(__name__)

CORS(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
config = {"apiKey": "AIzaSyCQlJNxtcp_ambo4mGWH9LxhK6Wsr3VlSM",
          "authDomain": "projectbase-1fca0.firebaseapp.com",
          "databaseURL": "https://projectbase-1fca0-default-rtdb.europe-west1.firebasedatabase.app",
          "projectId": "projectbase-1fca0",
          "storageBucket": "projectbase-1fca0.appspot.com",
          "messagingSenderId": "821113244030",
          "appId": "1:821113244030:web:1f86f63dfbba3d08c4cb2f",
          "measurementId": "G-J76JKQ1XX5"}
firebase = pyrebase.initialize_app(config)
db = firebase.database()

def handledate(date):
    date=str(date).split(' ')[0]
    return date

@app.route('/')
def hello_world():
    return 'Hello World!'
#fonction qui fixe la forme de la date from "Jan 26,2020" to "datetime")
def fixdate(date):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    first = date.split(' ')
    for i in range(len(months)):
        if first[0] == months[i]:
            first[0] = str(i + 1)
    first[1] = first[1].replace(',', '')
    date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
    return datetime(date[0], date[1], date[2])


#fonction qui transforme une serie temporelle vers une data supervisé ( x et y )
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values




def comparedate(new):
    date=new.split(' : ')[1]
    date=date.split('/')
    date=datedatedate.date(int('20'+date[2]),int(date[1]),int(date[0]))
    a_month = dateutil.relativedelta.relativedelta(months=4)
    return datedatedate.date.today() - a_month > date

@app.route('/scrap', methods=['POST', 'GET'])
def scrap():
    if request.method == 'GET':

        dic = dict()

        dic['company'] = []
        dic['last'] = []
        dic['chg'] = []
        dic['chgperc'] = []
        dic['date'] = []
        dic['type'] = []
        dic['links'] = []
        dic['symbols'] = []
        dic['news'] = []
        dic['volume'] = []

        try:
            dic = dict()
            dic['company'] = []
            dic['last'] = []

            dic['chgperc'] = []
            dic['date'] = []
            dic['type'] = []
            dic['links'] = []
            dic['symbols'] = []
            dic['news'] = []
            dic['volume'] = []
            dic['high'] = []
            dic['low'] = []
            dic['open'] = []

            url = 'https://www.investing.com/indices/tunindex'
            r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
            soup = BeautifulSoup(r.content, "html.parser")
            body = soup.find('body')
            divtuni = body.find('div', {'class': 'left current-data'})
            tuniprice = divtuni.find('span', {'class': 'arial_26 inlineblock pid-18823-last'}).text
            print('tuniprice' + str(tuniprice))
            tunichangeperc = divtuni.find('div', {'class': 'top bold inlineblock'}).find_all('span')[3].text
            print('tunichangeperc' + str(tunichangeperc))
            tuniopenlist = body.find('div', {'class': 'bottomText'}).find('ul').find_all('li')
            tuniopen = tuniopenlist[1].find('span', {'dir': 'ltr'}).text
            print('tuniopen' + str(tuniopen))
            tunivolume = body.find('div', {'class': 'bottomText'}).find('span',
                                                                        {'class': 'inlineblock pid-18823-volume'}).text
            print('tunivolume' + str(tunivolume))
            tunichange = divtuni.find('div', {'class': 'top bold inlineblock'}).find_all('span')[1].text
            for s in body.find('div', class_='inlineblock float_lang_base_1').find_all('td',
                                                                                       class_='left bold plusIconTd elp'):
                dic['type'].append('Gainer')
                dic['company'].append(s.text)

                dic['links'].append(s.find('a', href=True)['href'])

            for s in body.find('div', class_='inlineblock float_lang_base_2').find_all('td',
                                                                                       class_='left bold plusIconTd elp'):
                dic['company'].append(s.text)
                dic['links'].append(s.find('a', href=True)['href'])
                dic['type'].append('Loser')

            for s in body.find('table', class_='genTbl openTbl mostActiveStockTbl crossRatesTbl').find_all('td',
                                                                                                           class_='left bold plusIconTd'):
                dic['company'].append(s.text)
                dic['links'].append(s.find('a', href=True)['href'])
                dic['type'].append('Most Active')
            toremove = []

            for s in dic['links']:

                url = 'https://www.investing.com' + s

                r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
                soup = BeautifulSoup(r.content, "html.parser")
                body = soup.find('body')
                try:
                    h1 = body.find('h1', {
                        'class': 'text-2xl font-semibold instrument-header_title__GTWDv mobile:mb-2'}).text.split(' ')[
                        -1]

                except:
                    try:
                        h1 = body.find('h2', {'class': 'text-lg font-semibold'}).text.split(' ')[0]
                    except:
                        h1 = '-'

                h1 = h1.replace('(', '')
                h1 = h1.replace(')', '')
                dic['symbols'].append(h1)
                # ------------------------------------------endsymbol-----------------------------------------

                url = 'https://www.investing.com' + s + '-historical-data'

                r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
                soup = BeautifulSoup(r.content, "html.parser")
                body = soup.find('body')
                try:
                    tds = body.find('table', {'class': 'genTbl closedTbl historicalTbl'}).find_all('td')


                    dic['last'].append(tds[1].text)
                    dic['open'].append(tds[2].text)
                    dic['high'].append(tds[3].text)
                    dic['low'].append(tds[4].text)
                    dic['volume'].append(tds[5].text)
                    dic['chgperc'].append(tds[6].text)
                    dic['date'].append(tds[0].text)
                except:
                    toremove.append(s)
                    tds = body.find('table', {'class': 'genTbl closedTbl historicalTbl'}).find_all('td')
                    dic['date'].append('-')
                    dic['last'].append('-')
                    dic['open'].append('-')
                    dic['high'].append('-')
                    dic['low'].append('-')
                    dic['volume'].append('-')
                    dic['chgperc'].append('-')

            lst = ['type', 'company', 'links', 'symbols', 'date', 'last', 'open', 'high', 'low', 'volume', 'chgperc']
            print(toremove)
            print(len(toremove))
            print('lendic')
            for k in dic.keys():
                print(k)
                print(len(dic[k]))

            if len(toremove) >= 1:
                df = pd.DataFrame()
                for k in dic.keys():
                    if k != 'news':
                        df[k] = dic[k]
                for k in toremove:
                    print(k)
                    df = df[df['links'] != k]
                dic = dict()
                print('i removed bad links')
                for k in df.keys():
                    dic[k] = []
                print('show dataframe len')
                for k in df.keys():
                    print(k)
                    print(len(df[k]))
                df = df.reset_index(drop=True)
                try:
                    for i in range(len(df['links'])):
                        print(str(i) + '---------------------')
                        for k in df.keys():
                            print(k)
                            dic[k].append(df[k][i])
                except Exception as e:
                    print(e)
                dic['news'] = []
                print('im here')

                # -----------------------------------------------------startnews----------------------------------
            ###############################save todays list ################################

            session['todayslist']=dic['symbols']


            ##################################################################################
            for t in dic['symbols']:
                """
                dic['news'].append('-')
                """
                """
                """
                url = 'https://www.ilboursa.com/marches/cotation.aspx?s=' + t
                r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
                soup = BeautifulSoup(r.content, "html.parser")
                body = soup.find('body')
                news = []


                try:

                    dv1 = body.find('div', {'class': 'mobpad5'})
                    for new, date in zip(dv1.find_all('a'), dv1.find_all('span', {'class': 'sp1'})):
                        try:

                            news.append([new.text, date.text])


                        except:

                            kkkkk = 15

                    dic['news'].append(news)
                except:

                    dic['news'].append(['no news'])
                """
                """
            print('dic len ---------------------------')
            for k in dic.keys():
                print(len(dic[k]))

            df = pd.DataFrame()
            for key in dic.keys():
                if key != 'links':
                    df[key] = dic[key]
            # ------------------------------------------------------remove double and stock-----------------------------------------------------
            for i in df.index:
                df = df.reset_index(drop=True)
                if i in df.index:
                    for j in df.index:
                        df = df.reset_index(drop=True)

                        if j in df.index:
                            if i != j and df['symbols'][i] == df['symbols'][j]:
                                df['type'][i] = df['type'][i] + '-' + df['type'][j]
                                df = df.drop(df.index[j])


                        else:
                            continue

                else:
                    continue
            for symb in df['symbols']:


                thenewdic = dict()
                print(symb.upper().strip())
                price = df['last'][df['symbols'] == symb].values[0]
                vol = df['volume'][df['symbols'] == symb].values[0]
                change = df['chgperc'][df['symbols'] == symb].values[0]
                date = str(df['date'][df['symbols'] == symb].values[0])

                low = df['low'][df['symbols'] == symb].values[0]
                high = df['high'][df['symbols'] == symb].values[0]
                opn = df['open'][df['symbols'] == symb].values[0]
                thenewdic = db.child('realhistorical').child(symb.upper()).get().val()
                thenewdic['price'].append(str(price))
                thenewdic['date'].append(str(date))
                thenewdic['change'].append(str(change))
                thenewdic['vol'].append(str(vol))
                thenewdic['high'].append(str(high))
                thenewdic['low'].append(str(low))
                thenewdic['open'].append(str(opn))

                new = df['news'][df['symbols'] == symb].values[0]
                try:
                    news = []
                    for n in new:
                        news.append(n[0].replace(',', '').replace(':','') + ' : ' + n[1].replace('\r\n\t\t\t\t\t\t', ''))

                    c = db.child('news').child(symb.upper().strip()).get().val()
                    if str(type(c))!="<class 'NoneType'>":
                        newset=set()
                        for r in c:
                            newset.add(r)
                        for d in news:
                            newset.add(d)
                        lst=[]
                        for e in newset:
                            if comparedate(e)==False:
                                lst.append(e)
                        db.child('news').child(symb.upper().strip()).set(lst)
                    else:
                        lst = []
                        for e in news:
                            if comparedate(e) == False:
                                lst.append(e)

                    db.child('news').child(symb.upper().strip()).set(lst)
                except Exception as e:
                    db.child('news').child(symb.upper().strip()).set(['no news'])

                try:
                    file_name = symb+".pkl"

                    open_file = open('stockmodelsxgboost/' + file_name, "rb")
                    model = pickle.load(open_file)
                    open_file.close()



                    thedic=db.child('realhistorical').child(symb.upper().strip()).get().val()
                    newdf=pd.DataFrame()
                    for k in thedic.keys():
                        newdf[k]=thedic[k]
                    newdf['price'] = newdf['price'].apply(float)

                    lst = []
                    for l in newdf['price']:
                        lst.append(l)

                    newdf['price'] = lst
                    lst2 = []
                    for l in newdf['date']:
                        lst2.append(l)

                    newdf['date'] = lst2
                    newdf['date'] = newdf['date'].apply(fixdate)
                    newdf = newdf.set_index('date')
                    newdf['price'] = newdf['price'].apply(float)
                    newdf = newdf.drop(columns=['change', 'high', 'low', 'open', 'vol'])
                    data = series_to_supervised(newdf[-20:].values, n_in=19)

                    # split dataset

                    # seed history with training dataset
                    history = [x for x in data]
                    d = history[0]
                    d = d.tolist()
                    d.append(float(price))
                    d = np.array(d)
                    history[0] = d

                    print('this is history',history)
                    train = np.asarray(history)
                    print('this is train',train)
                    trainX, trainy = train[:, :-1], train[:, -1]

                    print('this is trainX',trainX)
                    print('this is trainy', trainy)

                    print('training '+symb+' model --------------')


                    model.fit(trainX,trainy)
                    open_file = open('stockmodelsxgboost/' + file_name, "wb")
                    pickle.dump(model, open_file)
                    open_file.close()
                except Exception as e:
                    print(e)









                """
                db.child('realhistorical').child(symb.upper().strip()).set(thenewdic)
                """


            # ----------------------------------------------------------------------------------
            dicdic = dict()
            for k in df.keys():
                dicdic[k] = []
            for i in range(len(df['symbols'])):
                for k in df.keys():
                    dicdic[k].append(str(df[k][i]))

            # return dic
            dicdic['tuniprice'] = []
            dicdic['tuniopen'] = []
            dicdic['tunichangeperc'] = []
            dicdic['tunivolume'] = []
            dicdic['tunichange'] = []
            for i in range(len(dicdic['symbols'])):
                dicdic['tuniprice'].append(tuniprice)
            dicdic['tuniprice'][0]

            for i in range(len(dicdic['symbols'])):
                dicdic['tuniopen'].append(tuniopen)
            dicdic['tuniopen'][0]
            for i in range(len(dicdic['symbols'])):
                dicdic['tunichangeperc'].append(tunichangeperc)
            dicdic['tunichangeperc'][0]
            for i in range(len(dicdic['symbols'])):
                dicdic['tunivolume'].append(tunivolume)
            for i in range(len(dicdic['symbols'])):
                dicdic['tunichange'].append(tunichange)
            print(dicdic['tunichange'][0])

            return dicdic
        except Exception as e:
            print('this is scrap error: '+str(e))
            print('except time')
            dic = dict()
            dic['company'] = []
            dic['last'] = []
            dic['chg'] = []
            dic['chgperc'] = []
            dic['closeyes'] = []
            dic['high'] = []
            dic['low'] = []
            dic['open'] = []
            dic['date'] = []
            dic['type'] = []
            dic['links'] = []
            dic['symbols'] = []
            dic['news'] = []
            dic['volume'] = []

            url = 'https://www.ilboursa.com/'
            r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
            soup = BeautifulSoup(r.content, "html.parser")
            body = soup.find('body')
            tds = body.find('div', {'class': 'bar12'}).find_all('td', {'class': 'arr_up'})
            for td in tds:
                dic['symbols'].append(td.find('a').text)
                dic['type'].append('Gainer')

            tds = body.find('div', {'class': 'bar13'}).find_all('td', {'class': 'arr_down'})
            for td in tds:
                dic['symbols'].append(td.find('a').text)
                dic['type'].append('Loser')

            for key in dic['symbols']:
                try:
                    url = 'https://www.ilboursa.com/marches/cotation.aspx?s=' + key
                    r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
                    soup = BeautifulSoup(r.content, "html.parser")
                    body = soup.find('body')
                    last = body.find('div', {'class': 'cot_v1b'}).text.replace('TND', '').strip().replace(',', '.')
                    company = body.find('h1', {'class': 'h1a mob26'}).text
                    try:
                        chgper = body.find('div', {'class': 'quote_up4'}).text
                    except:
                        chgper = body.find('div', {'class': 'quote_down4'}).text

                    date = str(datetime.now()).split(' ')[0]
                    link = '-'
                    symbol = key
                    volume = body.find('div', {'id': 'vol'}).text
                    news = '-'
                    tp = '-'
                    divopenhigh = body.find('div', {'class': 'cot_v21'}).find_all('div')
                    divcloselow = body.find('div', {'class': 'cot_v22'}).find_all('div')
                    closeyes = divcloselow[1].text
                    chg = str((float(chgper.strip().replace('+', '').replace('-', '').replace('%', '').replace(',',
                                                                                                               '.')) / 100) * float(
                    closeyes.strip().replace(',', '.')))
                    high = divopenhigh[3].text
                    low = divcloselow[3].text
                    opn = divopenhigh[1].text
                    dic['company'].append(company)
                    dic['last'].append(last)
                    dic['chg'].append(chg)
                    dic['chgperc'].append(chgper)
                    dic['date'].append(date)

                    dic['links'].append(link)

                    dic['news'].append(news)
                    dic['volume'].append(volume)
                    dic['closeyes'].append(closeyes)
                    dic['high'].append(high)
                    dic['low'].append(low)
                    dic['open'].append(opn)
                    print(key)
                except:
                    print(key)
                    print('error')
            return dic

if __name__ == '__main__':
    app.run(threaded=True)