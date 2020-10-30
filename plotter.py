import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pickle
import sqlite3


conn = sqlite3.connect('replaydb.sqlite')
cur = conn.cursor()

plt.figure()

cur.execute('SELECT * FROM perfs WHERE clientId == 0')

f = cur.fetchall()

tss = []
actions=[]

for row in f:
    ts, clientId, action = row[0], row[1], pickle.loads(row[2])
    # print(f'{ts},{clientId},{action[0]},{action[1]},{action[2]},{action[3]}')
    tss.append(ts)
    actions.append(action)

plt.figure()
print([eval(ac[0]) for ac in actions])
plt.plot(tss,[eval(ac[0]) for ac in actions], color='r')
plt.plot(tss,[eval(ac[1]) for ac in actions], color='b')
plt.show()