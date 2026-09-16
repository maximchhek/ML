import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import t
from numpy.linalg import solve

# Настройка графиков
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
np.random.seed(42)

# Генерируем данные для классификации и регрессии

X_class, y_class = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

print("Данные для классификации:")
print(f"X.shape: {X_class.shape}  (объекты × признаки)")
print(f"y.shape: {y_class.shape}  (метки классов)")
print(f"Уникальные классы: {np.unique(y_class)}")

X_reg, y_reg = make_regression(
    n_samples=100,
    n_features=1,
    noise=10.0,
    random_state=42
)

print("\nДанные для регрессии:")
print(f"X.shape: {X_reg.shape}")
print(f"y.shape: {y_reg.shape}")

# Визуализация наборов данных

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_class[y_class == 0, 0], X_class[y_class == 0, 1],
            color='blue', label='Класс 0', alpha=0.7)
ax1.scatter(X_class[y_class == 1, 0], X_class[y_class == 1, 1],
            color='red', label='Класс 1', alpha=0.7)
ax1.set_xlabel('Признак 1')
ax1.set_ylabel('Признак 2')
ax1.set_title('Задача классификации')
ax1.legend()
ax1.grid(True)

ax2.scatter(X_reg, y_reg, color='green', alpha=0.7)
ax2.set_xlabel('Признак X')
ax2.set_ylabel('Целевая переменная y')
ax2.set_title('Задача регрессии')
ax2.grid(True)

plt.tight_layout()
plt.savefig('fig_data.png', dpi=150)
plt.show()


# Основные операции с векторами

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

dot_product = np.dot(a, b)
print(f"Скалярное произведение a, b = {dot_product}")

euclidean_norm = np.sqrt(np.sum(a ** 2))
print(f"Евклидова норма a = {euclidean_norm:.4f}")

manhattan_norm = np.sum(np.abs(a))
print(f"Манхэттенская норма a = {manhattan_norm}")

cos_angle = dot_product / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Косинус угла между a и b = {cos_angle:.4f}")


# Расстояние Евклида для kNN

def euclidean_distance(x1, x2):
    diff = x1 - x2
    return np.sqrt(np.sum(diff ** 2))

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
expected = np.linalg.norm(x1 - x2)
print(f"Расстояние между x1 и x2: {euclidean_distance(x1, x2):.4f}")
print(f"Ожидаемое значение:        {expected:.4f}")


# Матрицы и линейные преобразования

X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
])
print("Матрица признаков X (4×3):")
print(X)
print(f"Размерность: {X.shape}")

w = np.array([0.5, 1.0, 2.0])
b = 1.5
predictions = X @ w + b
print(f"\nПредсказания модели: {predictions}")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"\nA + B =\n{A + B}")
print(f"A^T =\n{A.T}")
print(f"A × B =\n{A @ B}")


# Нормальное уравнение

X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])
y = np.array([2, 3, 4])

XTX = X.T @ X
print("X^T·X:")
print(XTX)

XTy = X.T @ y
print(f"X^T·y: {XTy}")

w = solve(XTX, XTy)
print(f"Вектор весов w = {w}")

predictions = X @ w
print(f"Предсказания: {predictions}")
print(f"Истинные y:    {y}")


# Срезы и индексация

data = np.random.rand(10, 5) * 100
print("Исходные данные:")
print(np.round(data, 2))

first_3 = data[:3, :]
print("\nПервые 3 объекта:")
print(np.round(first_3, 2))

selected_features = data[:, [0, 2, 4]]
print("\nПризнаки 1, 3, 5:")
print(np.round(selected_features, 2))

mask = data[:, 0] > 50
filtered = data[mask, :]
print(f"\nОбъектов с 1-м признаком > 50: {len(filtered)}")

N = len(data)
indices = np.random.permutation(N)
train_idx = indices[:int(0.7 * N)]
test_idx = indices[int(0.7 * N):]
X_train = data[train_idx, :]
X_test = data[test_idx, :]
print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки:  {len(X_test)}")


# Минимизация MSE

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + 1.0 + np.random.randn(100, 1) * 2.0

def mse_loss(params, X, y):
    w, b = params
    predictions = w * X.flatten() + b
    return np.mean((predictions - y.flatten()) ** 2)

initial_params = [0.0, 0.0]
result = minimize(mse_loss, initial_params, args=(X, y))

print("Результат оптимизации:")
print(f"  w = {result.x[0]:.4f}")
print(f"  b = {result.x[1]:.4f}")
print(f"  MSE = {result.fun:.4f}")

X_design = np.hstack([X, np.ones((len(X), 1))])
w_exact = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
print(f"\nТочное решение:")
print(f"  w = {w_exact[0, 0]:.4f}")
print(f"  b = {w_exact[1, 0]:.4f}")


# Логистическая регрессия через BCE

X_class_log, y_class_log = make_classification(
    n_samples=200, n_features=1,
    n_classes=2, n_informative=1,
    n_redundant=0, n_clusters_per_class=1,
    random_state=42
)

def logistic_loss(params, X, y):
    w, b = params
    z = w * X.flatten() + b
    p = 1 / (1 + np.exp(-z))
    loss = -np.mean(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    return loss

result = minimize(logistic_loss, [0.0, 0.0], args=(X_class_log, y_class_log))
print(f"w = {result.x[0]:.4f}, b = {result.x[1]:.4f}")
print(f"Минимальное значение BCE: {result.fun:.4f}")


# Статистический анализ

data = np.random.normal(loc=0, scale=1, size=1000)

mean = np.mean(data)
std = np.std(data)
median = np.median(data)
print(f"Среднее: {mean:.4f}")
print(f"Стандартное отклонение: {std:.4f}")
print(f"Медиана: {median:.4f}")

statistic, p_value = stats.shapiro(data[:100])
print(f"\nТест Шапиро-Уилка:")
print(f"  W-статистика = {statistic:.4f}")
print(f"  p-value      = {p_value:.4f}")
if p_value > 0.05:
    print("  данные можно считать нормально распределёнными")
else:
    print("  данные не распределены нормально")

confidence_level = 0.95
n = len(data)
se = std / np.sqrt(n)
t_critical = t.ppf((1 + confidence_level) / 2, df=n - 1)
ci_lower = mean - t_critical * se
ci_upper = mean + t_critical * se
print(f"\n95% доверительный интервал для среднего:")
print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")


# Визуализация kNN

X, y = make_classification(
    n_samples=200, n_features=2,
    n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1],
            color='blue', label='Класс 0', edgecolors='k', s=50)
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
            color='red', label='Класс 1', edgecolors='k', s=50)
plt.xlabel('Признак 1 (масштабированный)')
plt.ylabel('Признак 2 (масштабированный)')
plt.title('Разделяющая поверхность kNN (k=5)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fig_knn.png', dpi=150)
plt.show()


# Влияние параметра k

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
k_values_vis = [1, 3, 5, 10, 20, 50]

for ax, k in zip(axes.ravel(), k_values_vis):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_scaled, y)
    Z_k = knn_k.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z_k, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1],
               color='blue', s=20, edgecolors='k')
    ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
               color='red', s=20, edgecolors='k')
    ax.set_title(f'k = {k}')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle('Влияние параметра k на границу решения', fontsize=14)
plt.tight_layout()
plt.savefig('fig_k_analysis.png', dpi=150)
plt.show()


# Кривые обучения и переобучение

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

k_values = list(range(1, 21))
scores = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores[k] = (knn.score(X_train, y_train), knn.score(X_test, y_test))

train_scores = [scores[k][0] for k in k_values]
test_scores = [scores[k][1] for k in k_values]

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, 'o-', label='Обучающая выборка', linewidth=2)
plt.plot(k_values, test_scores, 's-', label='Тестовая выборка', linewidth=2)
plt.xlabel('k (количество соседей)')
plt.ylabel('Точность')
plt.title('Зависимость точности от параметра k')
plt.grid(True)
opt_k = k_values[int(np.argmax(test_scores))]
plt.axvline(x=opt_k, color='red', linestyle='--',
            label=f'Оптимальное k={opt_k}')
plt.legend()
plt.savefig('fig_learning_curve.png', dpi=150)
plt.show()

print("\nПроверка на переобучение")
for k in [1, 3, 5, 10, 15]:
    tr, te = scores[k]
    diff = tr - te
    print(f"k={k:2d}: Train={tr:.3f}, Test={te:.3f}, Разница={diff:.3f}")
print(f"\nОптимальное k = {opt_k}")


# Визуализация регрессии

np.random.seed(42)
X_reg = np.random.rand(100, 1) * 3
y_reg = 0.5 * X_reg.flatten() ** 2 - X_reg.flatten() + 2 + np.random.randn(100) * 0.2

lr = LinearRegression()
lr.fit(X_reg, y_reg)

poly = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('linear', LinearRegression())
])
poly.fit(X_reg, y_reg)

X_grid = np.linspace(0, 3, 100).reshape(-1, 1)
y_lr = lr.predict(X_grid)
y_poly = poly.predict(X_grid)

# Расчёт MSE для проверки эффекта переобучения
y_lr_train = lr.predict(X_reg)
mse_lr_train = np.mean((y_reg - y_lr_train) ** 2)
mse_lr_test = np.mean((lr.predict(X_grid) - (0.5 * X_grid.flatten()**2
                                            - X_grid.flatten() + 2)) ** 2)

y_poly_train = poly.predict(X_reg)
mse_poly_train = np.mean((y_reg - y_poly_train) ** 2)
mse_poly_test = np.mean((poly.predict(X_grid) - (0.5 * X_grid.flatten()**2
                                                - X_grid.flatten() + 2)) ** 2)

print("MSE на обучающей выборке и на истинной зависимости:")
print(f"  Линейная модель:  train={mse_lr_train:.4f}, test={mse_lr_test:.4f}")
print(f"  Полином 15 ст.:   train={mse_poly_train:.4f}, test={mse_poly_test:.4f}")

plt.figure(figsize=(12, 5))
plt.scatter(X_reg, y_reg, color='gray', alpha=0.7, label='Данные')
plt.plot(X_grid, y_lr, 'b-', linewidth=2, label='Линейная регрессия (недообучение)')
plt.plot(X_grid, y_poly, 'r-', linewidth=2, label='Полином 15 степени (переобучение)')
plt.xlabel('Признак X')
plt.ylabel('Целевая переменная y')
plt.legend()
plt.grid(True)
plt.title('Сравнение недообучения и переобучения')
plt.savefig('fig_regression.png', dpi=150)
plt.show()


# Реализация LinearRegressionGD

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias

            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

# Генерация данных
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Базовое обучение
model = LinearRegressionGD(learning_rate=0.1, n_iterations=500)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"Среднеквадратичная ошибка: {mse:.4f}")
print(f"Коэффициент детерминации R²: {r2:.4f}")

# График сходимости
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history)
plt.xlabel('Итерация')
plt.ylabel('MSE')
plt.title('Сходимость градиентного спуска')
plt.grid(True)
plt.yscale('log')
plt.savefig('fig_gd_convergence.png', dpi=150)
plt.show()

# Эксперимент с разными lr
print("\nЭксперимент с разными lr:")
for lr_val in [0.001, 0.01, 0.1, 0.5, 1.0]:
    m = LinearRegressionGD(learning_rate=lr_val, n_iterations=100)
    m.fit(X_train_scaled, y_train)
    final_loss = m.loss_history[-1]
    print(f"  lr={lr_val:5.3f}: финальный MSE = {final_loss:.4f}")


# Сравнение с sklearn

sk_lr = LinearRegression()
sk_lr.fit(X_train_scaled, y_train)
sk_pred = sk_lr.predict(X_test_scaled)
sk_mse = np.mean((y_test - sk_pred) ** 2)

print(f"MSE (градиентный спуск): {mse:.4f}")
print(f"MSE (Sklearn):           {sk_mse:.4f}")
print(f"Разница:                 {abs(mse - sk_mse):.6f}")