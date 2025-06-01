import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap

# ПАРАМЕТРИ ЗАДАЧІ
# Мода 1: wavelength = 10 (34.3 Гц)
# Мода 2: wavelength = 5 (68.6 Гц)
# Мода 3: wavelength = 3.33 (103 Гц)
# Мода 4: wavelength = 2.5 (137.2 Гц)
wavelength = 5  # довжина хвилі в метрах

# Фізичні константи
c = 343  # швидкість звуку в повітрі(м/с)
rho = 1.2  # густина повітря (кг/м³)
frequency = c / wavelength  # частота (Гц)
omega = 2 * np.pi * frequency  # кутова частота (рад/с)
k = omega / c  # хвильове число (рад/м)

# Геометрія кімнати (координати вершин в метрах)
room = np.array([[0, 0], [0, 5], [7, 5], [7, 4], [8, 3], [8, 2], [7, 1], [7, 0]])

# Параметри дискретизації
elements_per_wavelength = 6  # кількість елементів на довжину хвилі
mesh_size = wavelength / elements_per_wavelength  # розмір елементів сітки


def point_in_polygon(point, polygon):
    """Перевіряє, чи знаходиться точка всередині полігону"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def generate_mesh(vertices, h):
    """Генерує якісну трикутну сітку для полігональної області"""
    # Створюємо точки на границі з рівномірним кроком
    boundary_points = []
    n = len(vertices)

    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge_length = np.linalg.norm(v2 - v1)
        n_points = max(2, int(np.ceil(edge_length / h)))

        for j in range(n_points):
            t = j / n_points
            point = v1 + t * (v2 - v1)
            boundary_points.append(point)

    boundary_points = np.array(boundary_points)

    # Генеруємо регулярну сітку внутрішніх точок
    xmin, ymin = vertices.min(axis=0) - h / 2
    xmax, ymax = vertices.max(axis=0) + h / 2

    # Гексагональна упаковка для кращої якості елементів
    internal_points = []
    row = 0
    y = ymin + h
    while y < ymax:
        if row % 2 == 0:
            x = xmin + h
        else:
            x = xmin + h / 2

        while x < xmax:
            if point_in_polygon([x, y], vertices):
                # Перевіряємо відстань до границі
                min_dist = min(np.linalg.norm(boundary_points - [x, y], axis=1))
                if min_dist > h / 3:  # уникаємо занадто близьких точок
                    internal_points.append([x, y])
            x += h

        y += h * np.sqrt(3) / 2
        row += 1

    # Об'єднуємо всі точки
    if internal_points:
        all_points = np.vstack([boundary_points, np.array(internal_points)])
    else:
        all_points = boundary_points

    # Триангуляція Делоне
    tri = Delaunay(all_points)

    # Фільтруємо трикутники поза межами області
    valid_triangles = []
    for simplex in tri.simplices:
        centroid = all_points[simplex].mean(axis=0)
        if point_in_polygon(centroid, vertices):
            # Додаткова перевірка якості елемента
            triangle_points = all_points[simplex]
            area = 0.5 * abs(np.cross(triangle_points[1] - triangle_points[0],
                                      triangle_points[2] - triangle_points[0]))
            if area > (h ** 2) / 100:  # відкидаємо дуже малі елементи
                valid_triangles.append(simplex)

    return all_points, np.array(valid_triangles)


def compute_element_matrices(vertices):
    """Обчислює локальні матриці жорсткості та маси для трикутного елемента"""
    # Координати вершин
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    # Площа трикутника
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    # Коефіцієнти для градієнтів базисних функцій
    b1 = (y2 - y3) / (2 * area)
    b2 = (y3 - y1) / (2 * area)
    b3 = (y1 - y2) / (2 * area)

    c1 = (x3 - x2) / (2 * area)
    c2 = (x1 - x3) / (2 * area)
    c3 = (x2 - x1) / (2 * area)

    # Локальна матриця жорсткості
    B = np.array([[b1, b2, b3], [c1, c2, c3]])
    K_local = area * (B.T @ B)

    # Локальна матриця маси (точне інтегрування)
    M_local = area / 12 * np.array([[2, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 2]])

    return K_local, M_local


def assemble_system(points, triangles, k):
    """Збирає глобальні матриці системи для рівняння Гельмгольца"""
    n_points = len(points)

    # Використовуємо розріджений формат LIL для ефективної збірки
    K = lil_matrix((n_points, n_points), dtype=complex)
    M = lil_matrix((n_points, n_points), dtype=complex)

    print(f"Збірка матриць системи ({len(triangles)} елементів)...")

    for triangle in triangles:
        # Локальні матриці
        vertices = points[triangle]
        K_local, M_local = compute_element_matrices(vertices)

        # Додаємо до глобальних матриць
        for i in range(3):
            for j in range(3):
                K[triangle[i], triangle[j]] += K_local[i, j]
                M[triangle[i], triangle[j]] += M_local[i, j]

    # Матриця системи для рівняння Гельмгольца: (K - k²M)p = f
    A = K - k ** 2 * M

    return A.tocsr()  # конвертуємо в CSR для швидкого розв'язання


def solve_acoustic_problem(A, points, source_pos):
    """Розв'язує акустичну задачу з точковим джерелом"""
    n_points = len(points)

    # Вектор правої частини (точкове джерело)
    f = np.zeros(n_points, dtype=complex)

    # Знаходимо найближчий вузол до позиції джерела
    distances = np.linalg.norm(points - source_pos, axis=1)
    source_node = np.argmin(distances)

    # Амплітуда джерела
    f[source_node] = 1.0

    print("Розв'язання системи рівнянь...")
    # Розв'язуємо систему Ap = f
    pressure = spsolve(A, f)

    return pressure

# Генеруємо сітку
points, triangles = generate_mesh(room, mesh_size)
print(f"Кількість вузлів: {len(points)}")
print(f"Кількість елементів: {len(triangles)}")

# Збираємо систему
A = assemble_system(points, triangles, k)

# Позиція джерела звуку (приблизний центр кімнати)
source_position = np.array([4.0, 2.5])

# Розв'язуємо задачу
pressure = solve_acoustic_problem(A, points, source_position)

# ВІЗУАЛІЗАЦІЯ
# Створюємо власну кольорову карту для кращої візуалізації
colors = ['#0000ff', '#4040ff', '#8080ff', '#ffffff', '#ff8080', '#ff4040', '#ff0000']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('acoustic', colors, N=n_bins)

# Створюємо фігуру з двома підграфіками
fig = plt.figure(figsize=(16, 7))

# 1. Полігональна сітка
ax1 = fig.add_subplot(121)
ax1.triplot(points[:, 0], points[:, 1], triangles, 'k-', linewidth=0.3, alpha=0.6)
ax1.scatter(source_position[0], source_position[1],
            color='red', s=200, marker='*', edgecolor='black', linewidth=2,
            label='Джерело звуку', zorder=5)

# Контур кімнати
room_polygon = Polygon(room, fill=False, edgecolor='blue', linewidth=3,
                       label='Стіни кімнати')
ax1.add_patch(room_polygon)

ax1.set_xlabel('X (м)', fontsize=12)
ax1.set_ylabel('Y (м)', fontsize=12)
ax1.set_title(f'Полігональна сітка\n({len(triangles)} елементів, {len(points)} вузлів)',
              fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_aspect('equal')

# 2. Розподіл акустичного тиску
ax2 = fig.add_subplot(122)

# Створюємо триангуляцію для візуалізації
triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

# Візуалізуємо реальну частину тиску
pressure_real = np.real(pressure)
levels = np.linspace(-np.abs(pressure_real).max(), np.abs(pressure_real).max(), 50)

contourf = ax2.tricontourf(triang, pressure_real, levels=levels, cmap=cmap)
cbar = plt.colorbar(contourf, ax=ax2, label='Акустичний тиск (Па)', pad=0.01)

# Контурні лінії для виділення нулів (вузлів)
contour = ax2.tricontour(triang, pressure_real, levels=[0], colors='black',
                         linewidths=2, linestyles='--', alpha=0.8)

# Додаткові контури
ax2.tricontour(triang, pressure_real, levels=15, colors='gray',
               linewidths=0.5, alpha=0.4)

# Джерело та стіни
ax2.scatter(source_position[0], source_position[1],
            color='black', s=200, marker='*', edgecolor='white', linewidth=2,
            label='Джерело', zorder=5)

room_polygon2 = Polygon(room, fill=False, edgecolor='black', linewidth=3)
ax2.add_patch(room_polygon2)

ax2.set_xlabel('X (м)', fontsize=12)
ax2.set_ylabel('Y (м)', fontsize=12)
ax2.set_title(f'Розподіл акустичного тиску\nf = {frequency:.1f} Гц, λ = {wavelength} м',
              fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()

# ДОДАТКОВА ВІЗУАЛІЗАЦІЯ: АМПЛІТУДА ТА ВИЗНАЧЕННЯ ВУЗЛІВ/ПУЧНОСТЕЙ
fig2 = plt.figure(figsize=(10, 8))
ax3 = fig2.add_subplot(111)

# Амплітуда тиску
amplitude = np.abs(pressure)
max_amplitude = amplitude.max()

# Нормалізована амплітуда для кращої візуалізації
amplitude_norm = amplitude / max_amplitude

# Візуалізація амплітуди
levels_amp = np.linspace(0, 1, 30)
contourf_amp = ax3.tricontourf(triang, amplitude_norm, levels=levels_amp, cmap='hot')
cbar_amp = plt.colorbar(contourf_amp, ax=ax3, label='Нормалізована амплітуда')


# Легенда
ax3.scatter(source_position[0], source_position[1],
            color='green', s=200, marker='*', edgecolor='black', linewidth=2,
            label='Джерело', zorder=5)

# Контури для кращої візуалізації структури поля
ax3.tricontour(triang, amplitude_norm, levels=[0.2, 0.5, 0.8],
               colors='white', linewidths=1, alpha=0.5)

room_polygon3 = Polygon(room, fill=False, edgecolor='white', linewidth=3)
ax3.add_patch(room_polygon3)

ax3.set_xlabel('X (м)', fontsize=12)
ax3.set_ylabel('Y (м)', fontsize=12)
ax3.set_title(f'Амплітуда звукового поля з вузлами та пучностями\n' +
              f'Мода резонансу для λ = {wavelength} м (f = {frequency:.1f} Гц)',
              fontsize=14)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

plt.tight_layout()
plt.show()