import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Đọc dữ liệu từ file CSV
boston_data = pd.read_csv(r'D:\Py\HocMay\housing.csv', sep='\s+', header=None)

# Đặt tên cột cho DataFrame
boston_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Chọn các đặc trưng
features = ['CRIM', 'NOX', 'RM', 'AGE', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']
target = 'MEDV'

# Xử lý giá trị bị thiếu
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(boston_data[features])
y = boston_data[target].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Chia tập dữ liệu thành 80-20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hồi quy tuyến tính
poly = PolynomialFeatures(degree=2)  # Thêm các đặc trưng phi tuyến tính
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)
y_pred_lin_train = lin_reg.predict(X_train_poly)
y_pred_lin_test = lin_reg.predict(X_test_poly)

# Đánh giá hồi quy tuyến tính
mae_lin_train = mean_absolute_error(y_train, y_pred_lin_train)
mse_lin_train = mean_squared_error(y_train, y_pred_lin_train)
r2_lin_train = r2_score(y_train, y_pred_lin_train)

mae_lin_test = mean_absolute_error(y_test, y_pred_lin_test)
mse_lin_test = mean_squared_error(y_test, y_pred_lin_test)
r2_lin_test = r2_score(y_test, y_pred_lin_test)

print(f"Linear Regression (Train) - MAE: {mae_lin_train}, MSE: {mse_lin_train}, R-squared: {r2_lin_train}")
print(f"Linear Regression (Test) - MAE: {mae_lin_test}, MSE: {mse_lin_test}, R-squared: {r2_lin_test}")

# Cây quyết định
tree_reg = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1, min_samples_split=2, random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree_train = tree_reg.predict(X_train)
y_pred_tree_test = tree_reg.predict(X_test)

# Đánh giá cây quyết định
mae_tree_train = mean_absolute_error(y_train, y_pred_tree_train)
mse_tree_train = mean_squared_error(y_train, y_pred_tree_train)
r2_tree_train = r2_score(y_train, y_pred_tree_train)

mae_tree_test = mean_absolute_error(y_test, y_pred_tree_test)
mse_tree_test = mean_squared_error(y_test, y_pred_tree_test)
r2_tree_test = r2_score(y_test, y_pred_tree_test)

print(f"Decision Tree Regression (Train) - MAE: {mae_tree_train}, MSE: {mse_tree_train}, R-squared: {r2_tree_train}")
print(f"Decision Tree Regression (Test) - MAE: {mae_tree_test}, MSE: {mse_tree_test}, R-squared: {r2_tree_test}")

# Vẽ biểu đồ phân tán cho hồi quy tuyến tính
plt.figure(figsize=(15, 5))
# Biểu đồ cho tập train
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_lin_train, color='blue', label='Linear Regression (Train)')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=3)
plt.xlabel("Giá trị thực (Train)")
plt.ylabel("Giá trị dự đoán")
plt.title("Biểu đồ phân tán - Linear Regression trên Tập Train")
plt.legend()
plt.text(0.05, 0.95, f'MAE: {mae_lin_train:.2f}\nMSE: {mse_lin_train:.2f}\nR²: {r2_lin_train:.2f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Biểu đồ cho tập test
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lin_test, color='blue', label='Linear Regression (Test)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel("Giá trị thực (Test)")
plt.ylabel("Giá trị dự đoán")
plt.title("Biểu đồ phân tán - Linear Regression trên Tập Test")
plt.legend()
plt.text(0.05, 0.95, f'MAE: {mae_lin_test:.2f}\nMSE: {mse_lin_test:.2f}\nR²: {r2_lin_test:.2f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.tight_layout()
plt.show()

# Vẽ biểu đồ phân tán cho cây quyết định
plt.figure(figsize=(15, 5))
# Biểu đồ cho tập train
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_tree_train, color='green', label='Decision Tree (Train)')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=3)
plt.xlabel("Giá trị thực (Train)")
plt.ylabel("Giá trị dự đoán")
plt.title("Biểu đồ phân tán - Decision Tree trên Tập Train")
plt.legend()
plt.text(0.05, 0.95, f'MAE: {mae_tree_train:.2f}\nMSE: {mse_tree_train:.2f}\nR²: {r2_tree_train:.2f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Biểu đồ cho tập test
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tree_test, color='green', label='Decision Tree (Test)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel("Giá trị thực (Test)")
plt.ylabel("Giá trị dự đoán")
plt.title("Biểu đồ phân tán - Decision Tree trên Tập Test")
plt.legend()
plt.text(0.05, 0.95, f'MAE: {mae_tree_test:.2f}\nMSE: {mse_tree_test:.2f}\nR²: {r2_tree_test:.2f}', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.tight_layout()
plt.show()

# Vẽ biểu đồ so sánh giữa hai mô hình
plt.figure(figsize=(8, 6))
plt.bar(['Hồi quy tuyến tính', 'Cây quyết định'], 
        [mae_lin_test, mae_tree_test], color=['blue', 'green'])
plt.ylabel("MAE")
plt.title("So sánh MAE giữa Hồi quy tuyến tính và Cây quyết định")
plt.ylim(0, max(mae_lin_test, mae_tree_test) * 1.1)  # Tăng thêm một chút khoảng trống cho biểu đồ
plt.show()
