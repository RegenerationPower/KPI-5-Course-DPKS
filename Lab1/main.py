import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall

PROCESSORS_IN_CLUSTER = 6


def create_adjacency_matrix(num_clusters):
    num_processors = PROCESSORS_IN_CLUSTER * num_clusters
    adjacency_matrix = np.zeros((num_processors, num_processors), dtype=int)

    for cluster in range(0, num_clusters):
        base = cluster * PROCESSORS_IN_CLUSTER
        # Створюємо зв'язки між процесорами в межах одного кластера
        adjacency_matrix[base, base + 2] = 1  # 1-3 чорний
        adjacency_matrix[base + 1, base + 2] = 1  # 2-3 чорний
        adjacency_matrix[base + 2, base + 3] = 1  # 3-4 чорний
        adjacency_matrix[base + 3, base + 4] = 1  # 4-5 чорний
        adjacency_matrix[base + 3, base + 5] = 1  # 4-6 чорний

        if cluster >= 1:
            # Зв'язок між первинним кластером і вторинним кластером
            adjacency_matrix[0, base] = 1  # Первинний 1 з вторинним 1 синій
            adjacency_matrix[1, base + 1] = 1  # Первинний 2 з вторинним 2 світло-зелений
            adjacency_matrix[2, base + 2] = 1  # Первинний 3 з вторинним 3 жовтий
            adjacency_matrix[3, base + 3] = 1  # Первинний 4 з вторинним 4 бірюзовий
            adjacency_matrix[4, base + 4] = 1  # Первинний 5 з вторинним 5 червоний
            adjacency_matrix[5, base + 5] = 1  # Первинний 6 з вторинним 6 темно-зелений

            # Створюємо нерегулярний зв'язок між вторинними кластерами
            try:
                adjacency_matrix[base, base + PROCESSORS_IN_CLUSTER + 1] = 1  # червоний пунктир
            except IndexError:
                if cluster >= 2:
                    adjacency_matrix[base, PROCESSORS_IN_CLUSTER + 1] = 1

        # Створюємо нерегулярні зв'язки між вторинними кластерами
        if cluster >= 2:
            try:
                adjacency_matrix[base + 4, base + PROCESSORS_IN_CLUSTER + 4] = 1  # синій пунктир
            except IndexError:
                adjacency_matrix[base + 4, PROCESSORS_IN_CLUSTER + 4] = 1

            if cluster % 2 == 0:
                try:
                    adjacency_matrix[base + 2, base + 2 * PROCESSORS_IN_CLUSTER + 2] = 1  # жовтий пунктир
                except IndexError:
                    if cluster != 2:
                        adjacency_matrix[base + 2, 2 * PROCESSORS_IN_CLUSTER + 2] = 1

                try:
                    adjacency_matrix[base + 3, base - PROCESSORS_IN_CLUSTER + 2] = 1
                    adjacency_matrix[base + 3, base + PROCESSORS_IN_CLUSTER + 2] = 1  # світло-зелений пунктир
                except IndexError:
                    adjacency_matrix[base + 3, PROCESSORS_IN_CLUSTER + 2] = 1

            if cluster % 2 == 1:
                try:
                    adjacency_matrix[base + 3, base + 2 * PROCESSORS_IN_CLUSTER + 3] = 1  # бірюзовий пунктир
                except IndexError:
                    adjacency_matrix[base + 3, PROCESSORS_IN_CLUSTER + 3] = 1

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
    cluster_offset = 10  # Зміщення для кожного нового кластера

    # Додаємо вершини та зв'язки
    for i in range(num_processors):
        graph.add_node(i + 1)  # Іменуємо вузли 1, 2, ...
    for i in range(num_processors):
        for j in range(i + 1, num_processors):
            if adjacency_matrix[i, j] == 1:
                graph.add_edge(i + 1, j + 1)

    pos[1] = (0, 2)
    pos[2] = (2, 2)
    pos[3] = (1, 1)
    pos[4] = (1, -1)
    pos[5] = (0, -2)
    pos[6] = (2, -2)

    if num_processors // 6 > 1:
        angle_offset = 2 * np.pi / (num_processors // 6 - 1)

        for cluster_num in range(1, num_processors // 6):
            angle = angle_offset * (cluster_num - 1)
            base_pos_x = np.cos(angle) * cluster_offset
            base_pos_y = np.sin(angle) * cluster_offset

            base_index = cluster_num * 6
            pos[base_index + 1] = (base_pos_x, base_pos_y + 2)
            pos[base_index + 2] = (base_pos_x + 2, base_pos_y + 2)
            pos[base_index + 3] = (base_pos_x + 1, base_pos_y + 1)
            pos[base_index + 4] = (base_pos_x + 1, base_pos_y - 1)
            pos[base_index + 5] = (base_pos_x, base_pos_y - 2)
            pos[base_index + 6] = (base_pos_x + 2, base_pos_y - 2)

    plt.figure(figsize=(12, 8))
    cluster_labels = {}
    cluster_positions = []

    for cluster_num in range(num_processors // 6):
        base_index = cluster_num * 6
        cluster_center_x, cluster_center_y = pos[base_index + 1]
        cluster_positions.append((cluster_center_x, cluster_center_y - 1.5))
        cluster_labels[base_index + 1] = cluster_num + 1

    for i, (x, y) in enumerate(cluster_positions):
        plt.text(x, y, f"Кластер {i + 1}", fontsize=9, ha="center", color="black")

    for cluster_num in range(num_processors // 6):
        base_index = cluster_num * 6
        internal_edges = [
            (base_index + 1, base_index + 3),
            (base_index + 2, base_index + 3),
            (base_index + 3, base_index + 4),
            (base_index + 4, base_index + 5),
            (base_index + 4, base_index + 6),
        ]
        nx.draw_networkx_edges(graph, pos, edgelist=internal_edges, edge_color="black", width=1)

    colors = ["blue", "lightgreen", "yellow", "cyan", "red", "darkgreen"]
    for i in range(6):
        edges = [(i + 1, i + 1 + j * 6) for j in range(1, num_processors // 6)]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=colors[i], width=2,
                               connectionstyle="arc3,rad=0.3",
                               arrows=True)

    even_clusters = [cluster_num for cluster_num in range(1, num_processors // 6, 2)]
    odd_clusters = [cluster_num for cluster_num in range(2, num_processors // 6, 2)]

    irregular_edges_blue = []

    if step == 3:
        irregular_edges_blue.append((11, 17))

    else:
        irregular_edges_blue = [(i * 6 + 5, (i + 1) * 6 + 5) for i in range(1, num_processors // 6 - 1)]

        if num_processors // 6 > 2:
            irregular_edges_blue.append(((num_processors // 6 - 1) * 6 + 5, 11))

    irregular_edges_green = []

    for cluster_num in even_clusters:
        base_index = cluster_num * 6
        prev_cluster = cluster_num - 1

        if prev_cluster in odd_clusters:
            prev_base_index = prev_cluster * 6
            irregular_edges_green.append((base_index + 3, prev_base_index + 4))

        next_cluster = cluster_num + 1

        if next_cluster in odd_clusters:
            next_base_index = next_cluster * 6
            irregular_edges_green.append((base_index + 3, next_base_index + 4))

    if odd_clusters and even_clusters:
        last_odd_cluster = odd_clusters[-1]
        last_odd_index = last_odd_cluster * 6

        if last_odd_cluster + 1 not in even_clusters:
            first_even_index = even_clusters[0] * 6
            irregular_edges_green.append((first_even_index + 3, last_odd_index + 4))

    irregular_edges_yellow = []

    if len(odd_clusters) == 2:
        base_index = odd_clusters[0] * 6
        next_base_index = odd_clusters[1] * 6

        if (base_index + 3 in pos) and (next_base_index + 3 in pos):
            irregular_edges_yellow.append((base_index + 3, next_base_index + 3))

    else:

        for i in range(len(odd_clusters)):
            base_index = odd_clusters[i] * 6
            next_base_index = odd_clusters[(i + 1) % len(odd_clusters)] * 6

            if (base_index + 3 in pos) and (next_base_index + 3 in pos):
                irregular_edges_yellow.append((base_index + 3, next_base_index + 3))

    irregular_edges_cyan = []

    if len(even_clusters) == 2:
        base_index = even_clusters[0] * 6
        next_base_index = even_clusters[1] * 6

        if (base_index + 4 in pos) and (next_base_index + 4 in pos):
            irregular_edges_cyan.append((base_index + 4, next_base_index + 4))

    else:

        for cluster_num in even_clusters:
            base_index = cluster_num * 6
            next_base_index = ((cluster_num + 2) % (num_processors // 6)) * 6

            if next_base_index == 0:
                next_base_index = 6

            if (base_index + 3 in pos) and (next_base_index + 4 in pos):
                irregular_edges_cyan.append((base_index + 4, next_base_index + 4))

    irregular_edges_red = []

    for cluster_num in range(1, num_processors // 6 - 1):

        base_index = cluster_num * 6
        next_base_index = (cluster_num + 1) * 6

        if (base_index + 1 in pos) and (next_base_index + 2 in pos):
            irregular_edges_red.append((base_index + 1, next_base_index + 2))

    if num_processors // 6 > 2:
        irregular_edges_red.append(((num_processors // 6 - 1) * 6 + 1, 8))

    nx.draw_networkx_edges(graph, pos, edgelist=irregular_edges_blue, edge_color="blue", style="dashed", width=1.5,
                           connectionstyle="arc3,rad=0.3", arrows=True)

    nx.draw_networkx_edges(graph, pos, edgelist=irregular_edges_green, edge_color="lightgreen", style="dashed",
                           width=1.5,
                           connectionstyle="arc3,rad=0.3", arrows=True)

    nx.draw_networkx_edges(graph, pos, edgelist=irregular_edges_yellow, edge_color="yellow", style="dashed", width=1.5,
                           connectionstyle="arc3,rad=0.3", arrows=True)

    nx.draw_networkx_edges(graph, pos, edgelist=irregular_edges_cyan, edge_color="cyan", style="dashed", width=1.5,
                           connectionstyle="arc3,rad=0.3", arrows=True)

    nx.draw_networkx_edges(graph, pos, edgelist=irregular_edges_red, edge_color="red", style="dashed", width=1.5,
                           connectionstyle="arc3,rad=0.3", arrows=True)

    nx.draw_networkx_nodes(graph, pos, node_size=200, node_color="skyblue")
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")
    plt.title(f"Network Graph for Step {step}")
    plt.show()


def main():
    num_steps = int(input("Enter the number of clusters: "))  # Задаємо кількість кроків масштабування
    results = []
    final_adjacency_matrix = None

    for step in range(1, num_steps + 1):
        adjacency_matrix = create_adjacency_matrix(step)
        final_adjacency_matrix = adjacency_matrix
        properties = calculate_topological_properties(adjacency_matrix)
        results.append({
            "Step": step,
            "Properties": properties
        })

    print("Матриця суміжності з нумерацією від 1 до n (для останнього кроку):")
    print("    ", end="")  # Додатковий пробіл для вирівнювання
    for j in range(1, final_adjacency_matrix.shape[1] + 1):
        print(f"{j:2}", end=" ")
    print()

    for i in range(final_adjacency_matrix.shape[0]):
        print(f"{i + 1:2}  ", end="")  # Нумерація рядків і відступ
        for j in range(final_adjacency_matrix.shape[1]):
            print(f"{final_adjacency_matrix[i, j]:2}", end=" ")
        print()

    visualize_graph(final_adjacency_matrix, num_steps)

    final_result = results[-1]
    print("\nРезультати масштабування для останнього кроку:")
    print(f"Step {final_result['Step']}:")
    for prop, value in final_result["Properties"].items():
        print(f"  {prop}: {value}")


if __name__ == "__main__":
    main()
