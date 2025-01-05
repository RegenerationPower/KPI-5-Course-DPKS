import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall

PROCESSORS_IN_CLUSTER = 9


def create_adjacency_matrix(num_clusters):
    num_processors = PROCESSORS_IN_CLUSTER * num_clusters
    adjacency_matrix = np.zeros((num_processors, num_processors), dtype=int)
    grid_size = int(np.ceil(np.sqrt(num_clusters)))  # Визначення розміру решітки

    for cluster in range(0, num_clusters):
        base = cluster * PROCESSORS_IN_CLUSTER
        # Створюємо зв'язки між процесорами в межах одного кластера
        adjacency_matrix[base, base + 1] = 1  # 1-2 чорний
        adjacency_matrix[base, base + 3] = 1  # 1-4 чорний
        adjacency_matrix[base, base + 4] = 1  # 1-5 чорний
        adjacency_matrix[base + 1, base + 2] = 1  # 2-3 чорний
        adjacency_matrix[base + 2, base + 4] = 1  # 3-5 чорний
        adjacency_matrix[base + 2, base + 5] = 1  # 3-6 чорний
        adjacency_matrix[base + 3, base + 6] = 1  # 4-7 чорний
        adjacency_matrix[base + 4, base + 6] = 1  # 5-7 чорний
        adjacency_matrix[base + 4, base + 8] = 1  # 5-9 чорний
        adjacency_matrix[base + 5, base + 8] = 1  # 6-9 чорний
        adjacency_matrix[base + 6, base + 7] = 1  # 7-8 чорний
        adjacency_matrix[base + 7, base + 8] = 1  # 8-9 чорний

        if (cluster + 1) % grid_size != 0:
            try:
                adjacency_matrix[base + 2, base + PROCESSORS_IN_CLUSTER + 6] = 1  # Сині суцільні
            except IndexError:
                pass

            try:
                adjacency_matrix[base + 8, base + PROCESSORS_IN_CLUSTER] = 1  # Темно-зелені суцільні
            except IndexError:
                pass

            try:
                adjacency_matrix[base + 5, base + PROCESSORS_IN_CLUSTER + 3] = 1  # Світло-зелені суцільні
            except IndexError:
                pass

            try:
                adjacency_matrix[base + 8, base + (grid_size + 1) * PROCESSORS_IN_CLUSTER] = 1  # Сині пунктирні
            except IndexError:
                pass

        if (cluster + 1) % grid_size != 1:
            try:
                adjacency_matrix[base + 6, base + (grid_size - 1) * PROCESSORS_IN_CLUSTER + 2] = 1  # Світло-зелені пунктирні
            except IndexError:
                pass

        try:
            adjacency_matrix[base + 7, base + grid_size * PROCESSORS_IN_CLUSTER + 1] = 1  # Бірюзові суцільні
        except IndexError:
            pass

        try:
            adjacency_matrix[base + 6, base + grid_size * PROCESSORS_IN_CLUSTER + 2] = 1  # Жовті суцільні
        except IndexError:
            pass

        try:
            adjacency_matrix[base + 8, base + grid_size * PROCESSORS_IN_CLUSTER] = 1  # Червоні суцільні
        except IndexError:
            pass

        try:
            adjacency_matrix[base + 3, base + grid_size * PROCESSORS_IN_CLUSTER + 3] = 1  # Жовті пунктирні
        except IndexError:
            pass

        try:
            adjacency_matrix[base + 5, base + grid_size * PROCESSORS_IN_CLUSTER + 5] = 1  # Бірюзові пунктирні
        except IndexError:
            pass

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


def visualize_graph(adjacency_matrix, step):
    graph = nx.Graph()
    num_processors = adjacency_matrix.shape[0]
    pos = {}
    num_clusters = num_processors // PROCESSORS_IN_CLUSTER
    grid_size = int(np.ceil(np.sqrt(num_clusters)))  # Визначення розміру решітки
    cluster_spacing = 10  # Відстань між кластерами в решітці

    # Додаємо вершини та зв'язки
    for i in range(num_processors):
        graph.add_node(i + 1)  # Іменуємо вузли 1, 2, ...
    for i in range(num_processors):
        for j in range(i + 1, num_processors):
            if adjacency_matrix[i, j] == 1:
                graph.add_edge(i + 1, j + 1)

    # Розміщуємо всі кластери у вигляді решітки
    for cluster_num in range(num_clusters):
        row = cluster_num // grid_size
        col = cluster_num % grid_size
        base_pos_x = col * cluster_spacing
        base_pos_y = -row * cluster_spacing

        # Розміщення вузлів кластеру у початковій позиції
        base_index = cluster_num * PROCESSORS_IN_CLUSTER
        initial_positions = [
            (base_pos_x - 1, base_pos_y + 1),
            (base_pos_x, base_pos_y + 1),
            (base_pos_x + 1, base_pos_y + 1),
            (base_pos_x - 1, base_pos_y),
            (base_pos_x, base_pos_y),
            (base_pos_x + 1, base_pos_y),
            (base_pos_x - 1, base_pos_y - 1),
            (base_pos_x, base_pos_y - 1),
            (base_pos_x + 1, base_pos_y - 1),
        ]

        for i, node_position in enumerate(initial_positions):
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
            (base_index + 1, base_index + 4),
            (base_index + 1, base_index + 5),
            (base_index + 2, base_index + 3),
            (base_index + 3, base_index + 5),
            (base_index + 3, base_index + 6),
            (base_index + 4, base_index + 7),
            (base_index + 5, base_index + 7),
            (base_index + 5, base_index + 9),
            (base_index + 6, base_index + 9),
            (base_index + 7, base_index + 8),
            (base_index + 8, base_index + 9),
        ]
        nx.draw_networkx_edges(graph, pos, edgelist=internal_edges, edge_color="black", width=1)

    # Додаємо нерегулярні зв'язки
    for cluster_num in range(num_clusters):
        row = cluster_num // grid_size
        col = cluster_num % grid_size

        # Визначаємо правого сусіда
        if col < grid_size - 1 and cluster_num + 1 < num_clusters:
            next_cluster = cluster_num + 1
            current_base = cluster_num * PROCESSORS_IN_CLUSTER
            next_base = next_cluster * PROCESSORS_IN_CLUSTER

            # Сині зв'язки (3-7)
            blue_edges = [(current_base + 3, next_base + 7)]
            nx.draw_networkx_edges(graph, pos, edgelist=blue_edges, edge_color="blue", width=2)

            # Світло-зелені зв'язки (6-4)
            light_green_edges = [(current_base + 6, next_base + 4)]
            nx.draw_networkx_edges(graph, pos, edgelist=light_green_edges, edge_color="lightgreen", width=2)

            # Темно-зелені зв'язки (9-1)
            dark_green_edges = [(current_base + 9, next_base + 1)]
            nx.draw_networkx_edges(graph, pos, edgelist=dark_green_edges, edge_color="darkgreen", width=2)

        # Визначаємо нижнього сусіда
        if row < grid_size - 1 and cluster_num + grid_size < num_clusters:
            bottom_cluster = cluster_num + grid_size
            current_base = cluster_num * PROCESSORS_IN_CLUSTER
            bottom_base = bottom_cluster * PROCESSORS_IN_CLUSTER

            # Жовті зв'язки (7-3)
            yellow_edges = [(current_base + 7, bottom_base + 3)]
            nx.draw_networkx_edges(graph, pos, edgelist=yellow_edges, edge_color="yellow", width=2)

            # Бірюзові зв'язки (8-2)
            turquoise_edges = [(current_base + 8, bottom_base + 2)]
            nx.draw_networkx_edges(graph, pos, edgelist=turquoise_edges, edge_color="cyan", width=2)

            # Червоні зв'язки (9-1)
            red_edges = [(current_base + 9, bottom_base + 1)]
            nx.draw_networkx_edges(graph, pos, edgelist=red_edges, edge_color="red", width=2)

            # Жовті пунктирні зв'язки (4-4)
            yellow_dashed_edges = [(current_base + 4, bottom_base + 4)]
            nx.draw_networkx_edges(graph, pos, edgelist=yellow_dashed_edges, edge_color="yellow", style="dashed",
                                   width=2, connectionstyle="arc3,rad=0.2", arrows=True)

            # Бірюзові пунктирні зв'язки (6-6)
            turquoise_dashed_edges = [(current_base + 6, bottom_base + 6)]
            nx.draw_networkx_edges(graph, pos, edgelist=turquoise_dashed_edges, edge_color="cyan", style="dashed",
                                   width=2, connectionstyle="arc3,rad=-0.2", arrows=True)

        # Визначаємо правого нижнього сусіда (по діагоналі)
        if row < grid_size - 1 and col < grid_size - 1 and cluster_num + grid_size + 1 < num_clusters:
            diagonal_bottom_right = cluster_num + grid_size + 1
            current_base = cluster_num * PROCESSORS_IN_CLUSTER
            diagonal_base = diagonal_bottom_right * PROCESSORS_IN_CLUSTER

            # Сині пунктирні зв'язки (9-1)
            blue_dashed_edges = [(current_base + 9, diagonal_base + 1)]
            nx.draw_networkx_edges(graph, pos, edgelist=blue_dashed_edges, edge_color="blue", style="dashed", width=2)

        # Визначаємо лівого нижнього сусіда (по діагоналі)
        if row < grid_size - 1 and col > 0 and cluster_num + grid_size - 1 < num_clusters:
            diagonal_bottom_left = cluster_num + grid_size - 1
            current_base = cluster_num * PROCESSORS_IN_CLUSTER
            diagonal_base = diagonal_bottom_left * PROCESSORS_IN_CLUSTER

            # Світло-зелені пунктирні зв'язки (7-3)
            light_green_dashed_edges = [(current_base + 7, diagonal_base + 3)]
            nx.draw_networkx_edges(graph, pos, edgelist=light_green_dashed_edges, edge_color="lightgreen",
                                   style="dashed", width=2)

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
