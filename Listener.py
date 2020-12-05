import requests, time, datetime, pickle


#Listen for the top 4 movers by volume
file = open('logs.txt', 'a')

with open('dict.pickle', 'rb') as pik:
    master = pickle.load(pik)

def go():
    global file

    today = datetime.date.today()
    master[today] = {}

    file.write(str(today))
    file.write('\n')
    res = requests.get('https://api.iextrading.com/1.0/tops')

    j = res.json()

    def sortKey(dicto):
        return dicto['volume']

    j.sort(key= sortKey, reverse= True)


    for stock in j[:10]:
        file.write(f'{stock["symbol"]}: {stock["volume"]}\n')
        master[today][stock['symbol']] = stock['volume']

    file.write('\n')

while True:
    time.sleep(20)
    now = datetime.datetime.today()
    minute = now.minute
    hour = now.hour
    if hour == 16 and minute == 0:
        print('Triggered!')
        go()
        with open('dict.pickle', 'wb') as pik:
            pickle.dump(master, pik)
        time.sleep(60)
