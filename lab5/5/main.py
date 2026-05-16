from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import sys
sys.path.append(r'F:\Системы ИИ')
import utility

data = utility.importCSV('eustock.csv', 1, delimiter=',')

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])

x = []
for i in range(len(data)):
    x.append([i])

y_dax = [row[0] for row in data]
y_smi = [row[1] for row in data]
y_cac = [row[2] for row in data]
y_ftse = [row[3] for row in data]

model_dax = LinearRegression()
model_smi = LinearRegression()
model_cac = LinearRegression()
model_ftse = LinearRegression()

model_dax.fit(x, y_dax)
model_smi.fit(x, y_smi)
model_cac.fit(x, y_cac)
model_ftse.fit(x, y_ftse)

y_common = []
for i in range(len(x)):
    y_common.append((y_dax[i] + y_smi[i] + y_cac[i] + y_ftse[i])/4)

model_common = LinearRegression()
model_common.fit(x, y_common)

pred_dax = model_dax.predict(x)
pred_smi = model_smi.predict(x)
pred_cac = model_cac.predict(x)
pred_ftse = model_ftse.predict(x)
pred_common = model_common.predict(x)


plt.figure(figsize=(12, 6))

plt.plot(x, pred_dax, 'b-', label='DAX')
plt.plot(x, pred_smi, 'r-', label='SMI')
plt.plot(x, pred_cac, 'y-', label='CAC')
plt.plot(x, pred_ftse, 'c-', label='FTSE')
plt.plot(x, pred_common, 'g-', label='Common')
plt.legend()

plt.title('Динамика котировок')
plt.savefig('t5')
plt.show()

b_dax = model_dax.coef_[0]
b_smi = model_smi.coef_[0]
b_cac = model_cac.coef_[0]
b_ftse = model_ftse.coef_[0]
b_common = model_common.coef_[0]

print("Коэффициенты динамики (b):")
print(f"DAX:  {b_dax:.4f}")
print(f"SMI:  {b_smi:.4f}")
print(f"CAC:  {b_cac:.4f}")
print(f"FTSE: {b_ftse:.4f}")
print(f"Common: {b_common:.4f}")