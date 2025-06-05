import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap

wavelength = 10  # довжина хвилі (м)

# Фізичні константи
c = 343  # швидкість звуку в повітрі (м/с)
k = 2 *np.pi / wavelength  # хвильове число (рад/м)
frequency = c / wavelength  # частота (Гц)

# Геометрія кімнати (координати вершин в метрах)
room = np.array([[0, 0], [0, 5], [7, 5], [7, 4], [8, 3], [8, 2], [7, 1], [7, 0]])

# Параметри дискретизації
elements_per_wavelength = 6  # кількість елементів на довжину хвилі
mesh_size = wavelength / elements_per_wavelength  # розмір елементів сітки

# Параметри сітки
refinement_radius = 2.0  # радіус зони подрібнення навколо джерела (м)
refinement_factor = 0.3  # коефіцієнт подрібнення (0.3 = 30% від базового розміру)


def point_in_polygon(point, polygon):
    """Перевіряє, чи знаходиться точка всередині полігону (метод променів)"""
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


def generate_adaptive_mesh(vertices, h_base, source_pos, refinement_radius, refinement_factor):
    """
    Генерує трикутну сітку з підвищеною щільністю біля джерела

    Parameters:
    - vertices: координати вершин кімнати
    - h_base: базовий розмір елементів
    - source_pos: позиція джерела звуку
    - refinement_radius: радіус зони подрібнення навколо джерела
    - refinement_factor: коефіцієнт подрібнення (менше = дрібніша сітка)
    """

    def get_local_mesh_size(point):
        """Визначає локальний розмір сітки залежно від відстані до джерела"""
        dist_to_source = np.linalg.norm(point - source_pos)
        if dist_to_source < refinement_radius:
            # Зменшення розміру елементів біля джерела
            factor = refinement_factor + (1 - refinement_factor) * (dist_to_source / refinement_radius)
            return h_base * factor
        return h_base

    # Створюємо точки на границі з адаптивним кроком
    boundary_points = []
    n = len(vertices)

    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge_length = np.linalg.norm(v2 - v1)

        # Визначаємо кількість сегментів на основі локального розміру сітки
        edge_center = (v1 + v2) / 2
        local_h = get_local_mesh_size(edge_center)
        n_points = max(2, int(np.ceil(edge_length / local_h)))

        for j in range(n_points):
            t = j / n_points
            point = v1 + t * (v2 - v1)
            boundary_points.append(point)

    boundary_points = np.array(boundary_points)

    # Генеруємо адаптивну сітку внутрішніх точок
    xmin, ymin = vertices.min(axis=0) - h_base / 2
    xmax, ymax = vertices.max(axis=0) + h_base / 2

    internal_points = []

    # Спочатку додаємо точки в зоні подрібнення біля джерела
    r = refinement_factor * h_base
    n_circles = int(refinement_radius / r)

    for circle in range(1, n_circles + 1):
        radius = circle * r
        n_points_circle = int(2 * np.pi * radius / r)
        for i in range(n_points_circle):
            angle = 2 * np.pi * i / n_points_circle
            x = source_pos[0] + radius * np.cos(angle)
            y = source_pos[1] + radius * np.sin(angle)
            if point_in_polygon([x, y], vertices):
                internal_points.append([x, y])

    # Додаємо регулярну сітку з адаптивним кроком
    y = ymin + h_base
    row = 0
    while y < ymax:
        if row % 2 == 0:
            x = xmin + h_base
        else:
            x = xmin + h_base / 2

        while x < xmax:
            point = np.array([x, y])
            if point_in_polygon(point, vertices):
                local_h = get_local_mesh_size(point)

                # Перевіряємо чи точка не занадто близько до існуючих
                if internal_points:
                    min_dist = min(np.linalg.norm(np.array(internal_points) - point, axis=1))
                    if min_dist > local_h * 0.7:
                        internal_points.append(point)
                else:
                    internal_points.append(point)

                x += local_h
            else:
                x += h_base

        y += h_base * np.sqrt(3) / 2
        row += 1

    # Об'єднуємо всі точки
    if internal_points:
        all_points = np.vstack([boundary_points, np.array(internal_points)])
    else:
        all_points = boundary_points

    # Додаємо точку джерела
    all_points = np.vstack([all_points, source_pos])

    # Триангуляція Делоне
    tri = Delaunay(all_points)

    # Фільтруємо трикутники поза межами області
    valid_triangles = []
    for simplex in tri.simplices:
        centroid = all_points[simplex].mean(axis=0)
        if point_in_polygon(centroid, vertices):
            triangle_points = all_points[simplex]
            area = 0.5 * abs(np.cross(triangle_points[1] - triangle_points[0],
                                      triangle_points[2] - triangle_points[0]))
            if area > (h_base ** 2) / 1000:  # відкидаємо дуже малі елементи
                valid_triangles.append(simplex)

    return all_points, np.array(valid_triangles)


def compute_element_matrices(vertices):
    """
    Обчислює локальні матриці жорсткості та маси для трикутного елемента

    БАЗИСНІ ФУНКЦІЇ:
    Для трикутних елементів використовуються лінійні базисні функції (P1):
    N₁(x,y) = (a₁ + b₁x + c₁y) / (2A)
    N₂(x,y) = (a₂ + b₂x + c₂y) / (2A)
    N₃(x,y) = (a₃ + b₃x + c₃y) / (2A)

    де A - площа трикутника, а коефіцієнти b та c визначаються з геометрії:
    b₁ = y₂ - y₃, c₁ = x₃ - x₂
    b₂ = y₃ - y₁, c₂ = x₁ - x₃
    b₃ = y₁ - y₂, c₃ = x₂ - x₁
    """
    # Координати вершин
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    # Довжини сторін трикутника
    a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # сторона між вершинами 1-2
    b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)  # сторона між вершинами 2-3
    c = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)  # сторона між вершинами 3-1

    # Півпериметр
    s = (a + b + c) / 2

    # Площа трикутника за формулою Герона
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))

    # Коефіцієнти для градієнтів базисних функцій
    # ∇N₁ = [b₁, c₁] / (2A)
    # ∇N₂ = [b₂, c₂] / (2A)
    # ∇N₃ = [b₃, c₃] / (2A)
    b1 = (y2 - y3) / (2 * area)
    b2 = (y3 - y1) / (2 * area)
    b3 = (y1 - y2) / (2 * area)

    c1 = (x3 - x2) / (2 * area)
    c2 = (x1 - x3) / (2 * area)
    c3 = (x2 - x1) / (2 * area)

    # Локальна матриця жорсткості
    # K_ij = ∫∫_Ω ∇Nᵢ · ∇Nⱼ dA = A * (bᵢbⱼ + cᵢcⱼ)
    B = np.array([[b1, b2, b3], [c1, c2, c3]])
    K_local = area * (B.T @ B)

    # Локальна матриця маси (точне інтегрування)
    # M_ij = ∫∫_Ω Nᵢ Nⱼ dA
    # Для лінійних базисних функцій це дає:
    M_local = area / 12 * np.array([[2, 1, 1],  # діагональ: ∫N²ᵢ = A/6
                                    [1, 2, 1],  # недіагональ: ∫NᵢNⱼ = A/12
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

    # Розв'язуємо систему Ap = f
    pressure = spsolve(A, f)

    return pressure


# ОСНОВНИЙ КОД

# Позиція джерела звуку (центр кімнати)
source_position = np.array([4.0, 2.5])

# Генеруємо сітку
print(f"Базовий розмір елементів: {mesh_size:.3f} м")
print(f"Зона подрібнення: радіус {refinement_radius} м навколо джерела")
print(f"Коефіцієнт подрібнення: {refinement_factor} (менше = дрібніша сітка)")

points, triangles = generate_adaptive_mesh(room, mesh_size, source_position,
                                           refinement_radius=refinement_radius,
                                           refinement_factor=refinement_factor)
print(f"Кількість вузлів: {len(points)}")
print(f"Кількість елементів: {len(triangles)}")

# Збираємо систему
A = assemble_system(points, triangles, k)

# Розв'язуємо задачу
pressure = solve_acoustic_problem(A, points, source_position)

# ВІЗУАЛІЗАЦІЯ

# Створюємо власну кольорову карту для кращої візуалізації
colors = ['#0000ff', '#4040ff', '#8080ff', '#ffffff', '#ff8080', '#ff4040', '#ff0000']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('acoustic', colors, N=n_bins)

# Створюємо фігуру з двома підграфіками
fig = plt.figure(figsize=(16, 7))

# 1. Полігональна сітка (mesh)
ax1 = fig.add_subplot(121)

# Візуалізація сітки з кольором залежно від розміру елементів
# Обчислюємо середній розмір кожного елемента
element_sizes = []
for tri in triangles:
    vertices = points[tri]
    # Периметр трикутника
    perimeter = (np.linalg.norm(vertices[1] - vertices[0]) +
                 np.linalg.norm(vertices[2] - vertices[1]) +
                 np.linalg.norm(vertices[0] - vertices[2]))
    element_sizes.append(perimeter / 3)

element_sizes = np.array(element_sizes)

# Кольорова карта для розмірів елементів
tripcolor = ax1.tripcolor(points[:, 0], points[:, 1], triangles,
                          facecolors=element_sizes, cmap='viridis_r', alpha=0.7)
cbar_mesh = plt.colorbar(tripcolor, ax=ax1, label='Середній розмір елемента (м)', pad=0.01)

# Контури сітки
ax1.triplot(points[:, 0], points[:, 1], triangles, 'k-', linewidth=0.3, alpha=0.6)

# Джерело та зона подрібнення
circle = plt.Circle(source_position, refinement_radius, fill=False,
                    edgecolor='red', linewidth=2, linestyle='--', alpha=0.5,
                    label=f'Зона подрібнення (r={refinement_radius}м)')
ax1.add_patch(circle)

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
ax2.set_title(f'Діаграма розподілу акустичного тиску\nf = {frequency:.1f} Гц, λ = {wavelength} м',
              fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()
