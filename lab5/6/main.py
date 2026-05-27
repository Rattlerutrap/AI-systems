from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importCSV('JohnsonJohnson.csv', 1, delimiter=',')

year = []
for i in range(1960, 1981):
    year.append([i])

income_avg = []
income_q1 = []
income_q2 = []
income_q3 = []
income_q4 = []

for i in range(len(data)):
    match (i%4):
        case(0):
            income_q1.append(float(data[i][1]))
        case(1):
            income_q2.append(float(data[i][1]))
        case(2):
            income_q3.append(float(data[i][1]))
        case(3):
            income_q4.append(float(data[i][1]))
            income_avg.append((float(data[i][1]) + float(data[i-1][1]) + float(data[i-2][1]) + float(data[i-3][1])) / 4)
    

model_q1 = LinearRegression()
model_q2 = LinearRegression()
model_q3 = LinearRegression()
model_q4 = LinearRegression()
model_avg = LinearRegression()

model_q1.fit(year, income_q1)
model_q2.fit(year, income_q2)
model_q3.fit(year, income_q3)
model_q4.fit(year, income_q4)
model_avg.fit(year, income_avg)

pred_q1 = model_q1.predict(year)
pred_q2 = model_q2.predict(year)
pred_q3 = model_q3.predict(year)
pred_q4 = model_q4.predict(year)
pred_avg = model_avg.predict(year)

plt.figure(figsize=(12, 6))
plt.plot(year, pred_q1, 'b-', linewidth=2, label='Q1')
plt.plot(year, pred_q2, 'r-', linewidth=2, label='Q2')
plt.plot(year, pred_q3, 'y-', linewidth=2, label='Q3')
plt.plot(year, pred_q4, 'c-', linewidth=2, label='Q4')
plt.plot(year, pred_avg, 'g-', linewidth=2, label='Average')
plt.legend()
plt.savefig('t6')
plt.show()

b_q1 = model_q1.coef_[0]
b_q2 = model_q2.coef_[0]
b_q3 = model_q3.coef_[0]
b_q4 = model_q4.coef_[0]
b_avg = model_avg.coef_[0]

print("Коэффициенты динамики (b):")
print(f"Q1:  {b_q1:.4f}")
print(f"Q2:  {b_q2:.4f}")
print(f"Q3:  {b_q3:.4f}")
print(f"Q4:  {b_q4:.4f}")
print(f"Average:  {b_avg:.4f}")

print(f'Predicts in 2016:\nQ1: {model_q1.predict([[2016]])}\nQ2: {model_q2.predict([[2016]])}\nQ2: {model_q3.predict([[2016]])}\nQ2: {model_q4.predict([[2016]])}\nAvg: {model_avg.predict([[2016]])}')