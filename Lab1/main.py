import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall

def create_adjacency_matrix(num_clusters):
    num_processors = 6 * num_clusters
    adjacency_matrix = np.zeros((num_processors, num_processors), dtype=int)

    # Регулярні зв’язки для центрального кластера (кластер 0)
    base = 0
    adjacency_matrix[base, base + 2] = 1  # 1-3
    adjacency_matrix[base + 1, base + 2] = 1  # 2-3
    adjacency_matrix[base + 2, base + 3] = 1  # 3-4
    adjacency_matrix[base + 3, base + 4] = 1  # 4-5
    adjacency_matrix[base + 3, base + 5] = 1  # 4-6

    # Регулярні зв’язки для дочірніх кластерів та їх підключення до центрального
    for i in range(1, num_clusters):
        base = i * 6
        # Створюємо зв'язки в межах дочірнього кластера
        adjacency_matrix[base, base + 2] = 1  # 1-3
        adjacency_matrix[base + 1, base + 2] = 1  # 2-3
        adjacency_matrix[base + 2, base + 3] = 1  # 3-4
        adjacency_matrix[base + 3, base + 4] = 1  # 4-5
        adjacency_matrix[base + 3, base + 5] = 1  # 4-6

        # Зв'язок між центральним кластером і дочірнім кластером
        adjacency_matrix[0, base] = 1  # Головний 1 з дочірнім 1
        adjacency_matrix[1, base + 1] = 1  # Головний 2 з дочірнім 2
        adjacency_matrix[2, base + 2] = 1  # Головний 3 з дочірнім 3
        adjacency_matrix[3, base + 3] = 1  # Головний 4 з дочірнім 4
        adjacency_matrix[4, base + 4] = 1  # Головний 5 з дочірнім 5
        adjacency_matrix[5, base + 5] = 1  # Головний 6 з дочірнім 6

    # Робимо матрицю симетричною, оскільки зв’язки двосторонні
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    return adjacency_matrix

# Функція для обчислення топологічних характеристик
def calculate_topological_properties(adjacency_matrix):
    shortest_paths = floyd_warshall(adjacency_matrix, directed=False)
    diameter = np.max(shortest_paths[shortest_paths != np.inf])
    average_diameter = np.mean(shortest_paths[shortest_paths != np.inf])
    degree = np.max(np.sum(adjacency_matrix, axis=1))
    cost = np.sum(adjacency_matrix) // 2
    traffic = np.sum(shortest_paths[shortest_paths != np.inf]) / (adjacency_matrix.shape[0] * (adjacency_matrix.shape[0] - 1))

    return {
        "Diameter": diameter,
        "Average Diameter": average_diameter,
        "Degree": degree,
        "Cost": cost,
        "Traffic": traffic
    }

def visualize_graph(adjacency_matrix, step):
    G = nx.Graph()
    num_processors = adjacency_matrix.shape[0]

    # Додаємо вершини та зв'язки
    for i in range(num_processors):
        G.add_node(i + 1)  # Іменуємо вузли 1, 2, ...
    for i in range(num_processors):
        for j in range(i + 1, num_processors):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i + 1, j + 1)

    pos = {}
    cluster_offset = 10  # Зміщення для кожного нового кластера

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

    for cluster_num in range(num_processors // 6):
        base_index = cluster_num * 6
        internal_edges = [
            (base_index + 1, base_index + 3),
            (base_index + 2, base_index + 3),
            (base_index + 3, base_index + 4),
            (base_index + 4, base_index + 5),
            (base_index + 4, base_index + 6),
        ]
        nx.draw_networkx_edges(G, pos, edgelist=internal_edges, edge_color="black", width=1)

    colors = ["blue", "lightgreen", "yellow", "cyan", "red", "darkgreen"]
    for i in range(6):
        edges = [(i + 1, i + 1 + j * 6) for j in range(1, num_processors // 6)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors[i], width=2, connectionstyle="arc3,rad=0.3",
                               arrows=True)

    # Нерегулярні зв'язки між вторинними кластерами (без зв’язків з центральним кластером)
    irregular_edges = [(i * 6 + 5, (i + 1) * 6 + 5) for i in range(1, num_processors // 6 - 1)]
    if num_processors // 6 > 2:
        irregular_edges.append(((num_processors // 6 - 1) * 6 + 5, 11))

    nx.draw_networkx_edges(G, pos, edgelist=irregular_edges, edge_color="blue", style="dashed", width=1.5,
                           connectionstyle="arc3,rad=0.3", arrows=True)

    # Додавання нових світло-зелених пунктирних зв'язків між непарними вторинними кластерами за шаблоном 4-3
    irregular_edges_green = []
    for cluster_num in range(1, num_processors // 6, 2):  # тільки непарні вторинні кластери
        base_index = cluster_num * 6
        if cluster_num > 1:
            irregular_edges_green.append((base_index + 4, (cluster_num - 1) * 6 + 3))  # 4-3 з попереднім кластером
        if cluster_num < (num_processors // 6 - 1):
            irregular_edges_green.append((base_index + 4, (cluster_num + 1) * 6 + 3))  # 4-3 з наступним кластером

    nx.draw_networkx_edges(G, pos, edgelist=irregular_edges_green, edge_color="lightgreen", style="dashed", width=1.5,
                           connectionstyle="arc3,rad=0.3", arrows=True)

    irregular_edges_yellow = []
    for cluster_num in range(1, num_processors // 6, 2):  # тільки непарні вторинні кластери
        base_index = cluster_num * 6
        next_base_index = ((cluster_num + 2) % (num_processors // 6)) * 6
        if (base_index + 3 in pos) and (next_base_index + 3 in pos):
            irregular_edges_yellow.append((base_index + 3, next_base_index + 3))


    nx.draw_networkx_edges(G, pos, edgelist=irregular_edges_yellow, edge_color="yellow", style="dashed", width=1.5,
                           connectionstyle="arc3,rad=0.3", arrows=True)

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    plt.title(f"Network Graph for Step {step}")
    plt.show()

# Основна функція для запуску процесу
def main():
    num_steps = 6  # Задаємо кількість кроків масштабування
    results = []

    for step in range(1, num_steps + 1):
        adjacency_matrix = create_adjacency_matrix(step)
        properties = calculate_topological_properties(adjacency_matrix)
        results.append({
            "Step": step,
            "Properties": properties
        })

        # Візуалізація графу для кожного кроку
        visualize_graph(adjacency_matrix, step)

    # Виводимо результати
    for result in results:
        print(f"Step {result['Step']}:")
        for prop, value in result["Properties"].items():
            print(f"  {prop}: {value}")
        print()

if __name__ == "__main__":
    main()
