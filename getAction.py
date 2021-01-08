import sqlite3
import pickle

conn = sqlite3.connect('replaydb.sqlite')
cur = conn.cursor()

cur.execute('SELECT * FROM perfs WHERE clientId == 0')
f = cur.fetchall()

outf=open('out_c1.txt', 'w')
for row in f:
    ts, clientId, action = row[0], row[1], pickle.loads(row[2])
    outf.write(f'{ts},{clientId},{action[0]},{action[1]},{action[2]},{action[3]}\n')
    #print(f'{ts},{clientId},{action[0]},{action[1]},{action[2]},{action[3]}')
outf.close()

cur.execute('SELECT * FROM perfs WHERE clientId == 1')
f = cur.fetchall()

outf=open('out_c2.txt','w')
for row in f:
    ts, clientId, action = row[0], row[1], pickle.loads(row[2])
    outf.write(f'{ts},{clientId},{action[0]},{action[1]},{action[2]},{action[3]}\n')
    #print(f'{ts},{clientId},{action[0]},{action[1]},{action[2]},{action[3]}')
outf.close()

cur.execute('SELECT * FROM perfs WHERE clientId == 2')
f = cur.fetchall()

outf=open('out_c3.txt','w')
for row in f:
    ts, clientId, action = row[0], row[1], pickle.loads(row[2])
    outf.write(f'{ts},{clientId},{action[0]},{action[1]},{action[2]},{action[3]}\n')
    #print(f'{ts},{clientId},{action[0]},{action[1]},{action[2]},{action[3]}')
outf.close()
