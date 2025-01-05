import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall
import math

PROCESSORS_IN_CLUSTER = 7
ADDITIONAL_ROTATION = -90  # Додатковий фіксований кут для повороту (в градусах)

def create_adjacency_matrix(num_clusters):
    num_processors = PROCESSORS_IN_CLUSTER * num_clusters
    adjacency_matrix = np.zeros((num_processors, num_processors), dtype=int)

    for cluster in range(0, num_clusters):
        base = cluster * PROCESSORS_IN_CLUSTER
        # Створюємо зв'язки між процесорами в межах одного кластера
        adjacency_matrix[base, base + 1] = 1  # 1-2 чорний
        adjacency_matrix[base + 1, base + 2] = 1  # 2-3 чорний
        adjacency_matrix[base, base + 3] = 1  # 1-4 чорний
        adjacency_matrix[base + 1, base + 3] = 1  # 2-4 чорний
        adjacency_matrix[base + 2, base + 3] = 1  # 3-4 чорний
        adjacency_matrix[base + 3, base + 4] = 1  # 4-5 чорний
        adjacency_matrix[base + 3, base + 5] = 1  # 4-6 чорний
        adjacency_matrix[base + 3, base + 6] = 1  # 4-7 чорний

        # Сусідні кластери за принципом n-n
        for i in range(0, 7):
            try:
                adjacency_matrix[base + i, base + PROCESSORS_IN_CLUSTER + i] = 1
            except IndexError:
                if cluster not in (0, 1):
                    adjacency_matrix[base + i, i] = 1

        # Сині пунктирні зв'язки 2-2
        if cluster % 2 == 0 and num_clusters != 3:
            try:
                adjacency_matrix[base + 1, base + 2 * PROCESSORS_IN_CLUSTER + 1] = 1
            except IndexError:
                if cluster not in (0, 2):
                    adjacency_matrix[base + 1, 1] = 1

        # Світло-зелені пунктирні зв'зки 6-6
        if cluster % 2 == 1:
            try:
                adjacency_matrix[base + 5, base + 2 * PROCESSORS_IN_CLUSTER + 5] = 1
            except IndexError:
                if cluster not in (1, 3):
                    adjacency_matrix[base + 5, PROCESSORS_IN_CLUSTER + 5] = 1

        # Жовті пунктирні 4-5
        try:
            adjacency_matrix[base + 3, base + PROCESSORS_IN_CLUSTER + 4] = 1
        except IndexError:
            if cluster not in (0, 1):
                adjacency_matrix[base + 3, 4] = 1

        # Бірюзові пунктирні 3-1
        try:
            adjacency_matrix[base + 2, base + PROCESSORS_IN_CLUSTER + 0] = 1
        except IndexError:
            if cluster not in (0, 1):
                adjacency_matrix[base + 2, 0] = 1

    # Робимо матрицю симетричною, оскільки зв’язки двосторонні
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    return adjacency_matrix


# Функція для обчислення топологічних характеристик
def calculate_topological_properties(adjacency_matrix):
    num_processors = adjacency_matrix.shape[0]
    shortest_paths = floyd_warshall(adjacency_matrix, directed=False)
    d = np.max(shortest_paths[shortest_paths != np.inf])
    ad = np.sum(shortest_paths[shortest_paths != np.inf]) / (num_processors * (num_processors - 1))
    s = np.max(np.sum(adjacency_matrix, axis=1))
    c = np.sum(adjacency_matrix) // 2
    t = (2 * ad) / s

    return {
        "Number of processors": num_processors,
        "D": d,
        "aD": ad,
        "S": s,
        "C": c,
        "T": t
    }

# Поворот кластерів до центру графа
def rotate_point(point, center, angle):
    angle_rad = math.radians(angle)
    x, y = point
    cx, cy = center
    new_x = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
    new_y = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)
    return new_x, new_y


def calculate_angle_to_center(center):
    cx, cy = center
    angle = math.degrees(math.atan2(-cy, -cx))
    return angle


def visualize_graph(adjacency_matrix, step):
    graph = nx.Graph()
    num_processors = adjacency_matrix.shape[0]
    pos = {}
    cluster_offset = 10  # Зміщення для кожного нового кластера
    num_clusters = num_processors // PROCESSORS_IN_CLUSTER

    # Додаємо вершини та зв'язки
    for i in range(num_processors):
        graph.add_node(i + 1)  # Іменуємо вузли 1, 2, ...
    for i in range(num_processors):
        for j in range(i + 1, num_processors):
            if adjacency_matrix[i, j] == 1:
                graph.add_edge(i + 1, j + 1)

    # Розміщуємо всі кластери рівномірно по колу
    angle_offset = 2 * np.pi / num_clusters

    for cluster_num in range(num_clusters):
        angle = angle_offset * cluster_num
        base_pos_x = np.cos(angle) * cluster_offset
        base_pos_y = np.sin(angle) * cluster_offset
        center = (base_pos_x, base_pos_y)

        # Розміщення вузлів кластеру у початковій позиції
        base_index = cluster_num * PROCESSORS_IN_CLUSTER
        initial_positions = [
            (base_pos_x - 1, base_pos_y + 1),
            (base_pos_x, base_pos_y + 1),
            (base_pos_x + 1, base_pos_y + 1),
            (base_pos_x, base_pos_y),
            (base_pos_x - 1, base_pos_y - 1),
            (base_pos_x, base_pos_y - 1),
            (base_pos_x + 1, base_pos_y - 1),
        ]

        rotation_angle = calculate_angle_to_center(center) + ADDITIONAL_ROTATION

        # Обертання вузлів кластеру навколо центру кластеру
        rotated_positions = [
            rotate_point(pos, center, rotation_angle)
            for pos in initial_positions
        ]

        for i, node_position in enumerate(rotated_positions):
            pos[base_index + i + 1] = node_position

    plt.figure(figsize=(12, 8))
    cluster_labels = {}
    cluster_positions = []

    for cluster_num in range(num_clusters):
        base_index = cluster_num * PROCESSORS_IN_CLUSTER
        cluster_center_x, cluster_center_y = pos[base_index + 1]
        cluster_positions.append((cluster_center_x, cluster_center_y - 1.5))
        cluster_labels[base_index + 1] = cluster_num + 1

    for i, (x, y) in enumerate(cluster_positions):
        plt.text(x, y, f"Кластер {i + 1}", fontsize=9, ha="center", color="black")

    for cluster_num in range(num_clusters):
        base_index = cluster_num * PROCESSORS_IN_CLUSTER
        internal_edges = [
            (base_index + 1, base_index + 2),
            (base_index + 2, base_index + 3),
            (base_index + 1, base_index + 4),
            (base_index + 2, base_index + 4),
            (base_index + 3, base_index + 4),
            (base_index + 4, base_index + 5),
            (base_index + 4, base_index + 6),
            (base_index + 4, base_index + 7),
        ]
        nx.draw_networkx_edges(graph, pos, edgelist=internal_edges, edge_color="black", width=1)

    colors = ["blue", "lightgreen", "yellow", "cyan", "red", "darkgreen", "pink"]

    if step < 3:
        for i in range(PROCESSORS_IN_CLUSTER):
            edges = [(i + 1, i + 1 + j * PROCESSORS_IN_CLUSTER) for j in range(1, num_clusters)]
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=colors[i], width=2,
                                   connectionstyle="arc3,rad=0.3", arrows=True)
    else:
        for cluster_num in range(num_clusters):
            next_cluster = (cluster_num + 1) % num_clusters
            for i in range(PROCESSORS_IN_CLUSTER):
                node_from = cluster_num * PROCESSORS_IN_CLUSTER + i + 1
                node_to = next_cluster * PROCESSORS_IN_CLUSTER + i + 1
                graph.add_edge(node_from, node_to)
                nx.draw_networkx_edges(graph, pos, edgelist=[(node_from, node_to)],
                                      edge_color=colors[i % len(colors)], width=2,
                                      connectionstyle="arc3,rad=0.3", arrows=True)

    # Додаємо нерегулярні зв'язки
    blue_edges = []
    green_edges = []
    yellow_edges = []
    cyan_edges = []

    for cluster_num in range(num_clusters):
        if cluster_num % 2 == 0:
            current_node = cluster_num * PROCESSORS_IN_CLUSTER + 2
            next_cluster = (cluster_num + 2) if (cluster_num + 2) < num_clusters else 0
            next_node = next_cluster * PROCESSORS_IN_CLUSTER + 2
            blue_edges.append((current_node, next_node))

        if step == 5:
            if cluster_num % 2 == 1:
                current_node = cluster_num * PROCESSORS_IN_CLUSTER + 6
                next_cluster = cluster_num + 2
                if next_cluster < num_clusters:
                    next_node = next_cluster * PROCESSORS_IN_CLUSTER + 6
                    green_edges.append((current_node, next_node))
        else:
            if cluster_num % 2 == 1:
                current_node = cluster_num * PROCESSORS_IN_CLUSTER + 6
                next_cluster = (cluster_num + 2) if (cluster_num + 2) < num_clusters else 1
                next_node = next_cluster * PROCESSORS_IN_CLUSTER + 6
                green_edges.append((current_node, next_node))

        current_node = cluster_num * PROCESSORS_IN_CLUSTER + 4
        next_cluster = (cluster_num + 1) % num_clusters
        next_node = next_cluster * PROCESSORS_IN_CLUSTER + 5
        yellow_edges.append((current_node, next_node))

        current_node = cluster_num * PROCESSORS_IN_CLUSTER + 3
        next_cluster = (cluster_num + 1) % num_clusters
        next_node = next_cluster * PROCESSORS_IN_CLUSTER + 1
        cyan_edges.append((current_node, next_node))

    nx.draw_networkx_edges(graph, pos, edgelist=blue_edges, edge_color="blue", style="dashed", width=2)
    nx.draw_networkx_edges(graph, pos, edgelist=green_edges, edge_color="lightgreen", style="dashed", width=2)
    nx.draw_networkx_edges(graph, pos, edgelist=yellow_edges, edge_color="yellow", style="dashed", width=2)
    nx.draw_networkx_edges(graph, pos, edgelist=cyan_edges, edge_color="cyan", style="dashed", width=2)

    nx.draw_networkx_nodes(graph, pos, node_size=200, node_color="skyblue")
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")
    plt.title(f"Network Graph for Step {step}")
    plt.show()


def main():
    try:
        num_steps = int(input("Введіть кількість кластерів: "))  # Задаємо кількість кроків масштабування
        if num_steps < 1:
            raise ValueError("Кількість кластерів повинна бути принаймні 1.")
    except ValueError as e:
        print(f"Неправильний ввід: {e}")
        return

    final_step = num_steps
    final_adjacency_matrix = create_adjacency_matrix(final_step)
    final_properties = calculate_topological_properties(final_adjacency_matrix)

    print("Матриця суміжності з нумерацією від 1 до n:")
    print("    ", end="")  # Додатковий пробіл для вирівнювання
    for j in range(1, final_adjacency_matrix.shape[1] + 1):
        print(f"{j:2}", end=" ")
    print()

    for i in range(final_adjacency_matrix.shape[0]):
        print(f"{i + 1:2}  ", end="")  # Нумерація рядків і відступ
        for j in range(final_adjacency_matrix.shape[1]):
            print(f"{final_adjacency_matrix[i, j]:2}", end=" ")
        print()

    visualize_graph(final_adjacency_matrix, final_step)

    print("\nРезультати масштабування:")
    print(f"Step {final_step}:")
    for prop, value in final_properties.items():
        print(f"  {prop}: {value}")


# def main():
#     try:
#         num_steps = int(input("Введіть кількість кластерів: "))  # Задаємо кількість кроків масштабування
#         if num_steps < 1:
#             raise ValueError("Кількість кластерів повинна бути принаймні 1.")
#     except ValueError as e:
#         print(f"Неправильний ввід: {e}")
#         return
#
#     results = []
#
#     for step in range(1, num_steps + 1):
#         print(f"\n--- Крок {step} ---")
#
#         # Генеруємо матрицю суміжності для поточного етапу
#         adjacency_matrix = create_adjacency_matrix(step)
#         properties = calculate_topological_properties(adjacency_matrix)
#         results.append({
#             "Step": step,
#             "Properties": properties
#         })
#
#         # Виводимо матрицю суміжності з нумерацією
#         print("Матриця суміжності з нумерацією від 1 до n:")
#         print("    ", end="")  # Додатковий пробіл для вирівнювання
#         for j in range(1, adjacency_matrix.shape[1] + 1):
#             print(f"{j:2}", end=" ")
#         print()
#
#         for i in range(adjacency_matrix.shape[0]):
#             print(f"{i + 1:2}  ", end="")  # Нумерація рядків і відступ
#             for j in range(adjacency_matrix.shape[1]):
#                 print(f"{adjacency_matrix[i, j]:2}", end=" ")
#             print()
#
#         visualize_graph(adjacency_matrix, step)
#
#         print("\nРезультати масштабування для цього етапу:")
#         print(f"Етап {step}:")
#         for prop, value in properties.items():
#             print(f"  {prop}: {value}")


if __name__ == "__main__":
    main()
