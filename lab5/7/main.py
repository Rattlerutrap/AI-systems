from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importCSV('cars.csv', 1, delimiter=',')

speed = []
dist = []
for i in data:
    speed.append([float(i[0])])
    dist.append(float(i[1]))

model = LinearRegression()
model.fit(speed, dist)

predict = model.predict(speed)

for i in range(len(predict)):
    print(f'{speed[i]}:{predict[i]}')

predict40 = model.predict([[40]])

print(f'40 mph: {predict40}')

plt.figure(figsize=(12,6))
plt.plot(speed, predict, color='b')
plt.plot(speed, dist, color='r')
plt.xlabel('Speed')
plt.ylabel('Distance')
plt.grid(visible=True)
plt.title('Dist on speed')
plt.savefig('t7')
plt.show()

