import sqlite3
import pickle

conn = sqlite3.connect('replaydb.sqlite')
cur = conn.cursor()

cur.execute('SELECT * FROM perfs WHERE clientId == 0')

f = cur.fetchall()

for row in f:
    ts, clientId, action = row[0], row[1], pickle.loads(row[2])
    print(f'{ts},{clientId},{action[0]},{action[1]},{action[2]},{action[3]}')
