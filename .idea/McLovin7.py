import random
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from scipy.stats import qmc
from math import sqrt
from scipy.special import erf


class PointGenerator():
    def __init__(self, height, width, rho, type):
        self.height = height
        self.width = width
        self.rho = rho
        self.type = type

    def gen(self):

        # Генерация случайным образом
        if self.type == 1:
            ar = []
            n = math.ceil(self.height * self.width * self.rho)
            for i in range(0, n):
                x = self.width * random.random() - self.width / 2
                y = self.height * random.random() - self.height / 2
                ar.append((x, y))
            return np.array(ar)

        # Генерация квадратной сетки
        if self.type == 2:
            ar = []
            n = math.ceil(self.height * self.width * self.rho)
            for i in range(0, n):
                x = self.width * random.random() - self.width / 2
                y = self.height * random.random() - self.height / 2
                ar.append((x, y))
            return np.array(ar)

        # Квазислучайная генерация
        if self.type == 3:
            n = math.ceil(self.height * self.width * self.rho)

            # Halton
            engine = qmc.Halton(2)
            sample = engine.random(n)
            return sample


class DeterminingTheEdgeWeight():
    def __init__(self, f, r):
        self.f = f
        self.r = r

    def p1(r, f):
        beta_f = (0.1 * f ** 2) / (1 + f ** 2) + (40 * f ** 2) / (4100 + f ** 2) + 2.75 * 10 ** (-4) * f ** 2 + 0.0003
        x = sqrt(f / 10) * ((6.71 * 10 ** 3) / r) * (10 ** (-0.05 * beta_f * 10 ** (-3) * r))
        return erf(x)


    def p2(r, f):
        gamma = (f / r ** 2) * 10 ** (0.1 * (DeterminingTheEdgeWeight.SNR(r, f) - DeterminingTheEdgeWeight.beta(f) *
                                             10 ** (-3) * r))
        qe = 0.5 * (1 - sqrt(gamma / (1 + gamma)))
        return (1 - qe) ** 256

    def beta(f):
        return (0.1 * f ** 2) / (1 + f ** 2) + (40 * f ** 2) / (4100 + f ** 2) + 2.75e-4 * f ** 2 + 0.0003

    def SNR(r, f):
        return 80


class Clustering():
    def __init__(self, Graph):
        self.G = Graph

    # Создает количество списков равному количеству референсов. В список добавляется по референсу
    # В каждый список добавляются соседи референсов
    # В каждый список добавляются соседи сенсоров, которые были добавлены в список
    # Сравнивается количество сенсоров в списках, удаляются смежные сенсоры у того списка, у кого больше сенсоров
    def createGroups(self):
        G = self.G
        referenses = []
        for node in G.nodes:
            if G.nodes[node]['object'] == 'R':
                referenses.append(node)
        print(referenses)

        groups = []
        for node in G.nodes:
            if G.nodes[node]['object'] == 'R':
                groups.append([node])

        # Добавляем соседей референсов в группы
        for group in groups:
            for neighbor in list(G.neighbors(group[0])):
                if (neighbor not in group) and (neighbor not in referenses):
                    group.append(neighbor)

        # Добавляем соседей сенсоров в группы
        for group in groups:
            for i in range(1, len(group)):
                for neighbor in list(G.neighbors(group[i])):
                    if (neighbor not in group) and (neighbor not in referenses):
                        group.append(neighbor)

        # Удаляем одинаковые сенсоры в группах, оставляя тот, у которого длина пути меньше
        for group in groups:
            for sensor in group: # Можно упростить, если начинать не с референса, но хз, как не выйти за пределы массива
                for other_group in groups:  # Для разбития на кластеры одинакового размера поменять последний and на
                                                                                        # len(group) >= len(other_group)
                    if (sensor in other_group) and (other_group != group) and (
                            nx.shortest_path_length(G, group[0], sensor, weight='dist') <=
                            nx.shortest_path_length(G, other_group[0], sensor, weight='dist')):
                        other_group.remove(sensor)
        return groups


    def graphDivision(self, groups):
        G = self.G
        g = []
        for i in range(0, len(groups)):
            g.append(G.subgraph(groups[i]))
        # Если хотим кластеры со всеми ребрами, return g после этой строки


        #unfrozenGraph
        unfrozen_graph = []
        # Удаляем лишние ребра
        for i in range(0, len(groups)):
            unfrozen_graph.append(nx.Graph())
            unfrozen_graph[i].add_nodes_from(g[i].nodes(data=True))
            edges_to_keep = Clustering.remove_extra_edges(g[i], groups[i])
            #print()
            #print('edges', edges_to_keep)
            if edges_to_keep != 0:
                for u, v in edges_to_keep:
                    #print(u, v)
                    if g[i].has_edge(u, v):
                        unfrozen_graph[i].add_edge(u, v, **g[i][u][v])
                # print(unfrozen_graph[i].edges(data=True))

        return unfrozen_graph

    def remove_extra_edges(g, group):
        arr = []
        if len(group) == 1:
            return 0
        reference = group[0]
        group.reverse()
        path = []
        for node in group:
            if node != reference:
                path.append(nx.shortest_path(g, node, reference))
        for pat in path:
            arr.append((pat[0], pat[1]))
        return arr


class GraphGenerator():
    def __init__(self, sample, number_of_references, dist):
        self.sample = sample
        self.number_of_references = number_of_references
        self.dist = dist

    # euc dist function
    def eucDist(p1, p2):
        #print(p1[0], p2[0])
        #print(p1[1], p2[1])
        #print()
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def abc(self):
        pos = dict(enumerate(self.sample, 0))
        #print(pos)
        G = nx.Graph()

        # generate nodes in graph
        nodes = [i for i in range(len(sample))]
        G.add_nodes_from(nodes)
        for node in nodes:
            G.nodes[node]['X'] = sample[node][0]
            G.nodes[node]['Y'] = sample[node][1]
            G.nodes[node]['data'] = random.uniform(*(0, 100))
            G.nodes[node]['charge'] = 864
            if node % self.number_of_references == 0:
                G.nodes[node]['object'] = 'R'
            else:
                G.nodes[node]['object'] = 'D'

        print(G.nodes(data=True))


        # generate edges in graph
        length = self.dist
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dist = GraphGenerator.eucDist(pos[i], pos[j]) * 10 ** 3
                #print(dist)
                if dist <= length:
                    p = DeterminingTheEdgeWeight.p2(dist, 40)
                    #print(p)
                    #print(dist)
                    G.add_edge(i, j, weight=p)
                    G[i][j]["dist"] = dist
        return G


class TransferData():
    def __init__(self, g, groups):
        self.g = g
        self.groups = groups

    def transfer(self):
        groups = self.groups
        g = self.g
        # Передаем данные в каждом кластере
        for i in range(0, len(groups)):
            TransferData.transfer_data(g[i], groups[i])

    def transfer_data(g, group):
        # Функция передачи данных
        if len(group) == 1:
            return 0
        reference = group[0]
        group.reverse()
        for node in group:
            if node != reference:
                path = nx.shortest_path(g, node, reference, weight='dist')
                g.nodes[path[path.index(node) + 1]]["data"] += g.nodes[node]["data"]
                g.nodes[node]["charge"] -= 0.5
                g.nodes[path[path.index(node) + 1]]["charge"] -= 0.5
                #print(g.nodes[node]["charge"])


class DrawGraph():
        def __init__(self, Graph, sample):
            self.G = Graph
            self.sample = sample

        def drawTheEntireGraph(self):
            G = self.G
            pos = dict(enumerate(self.sample, 0))

            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
            esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

            # draw nodes
            node_color = 'skyblue'
            node_colors = []
            for node in G.nodes:
                if G.nodes[node]['object'] == 'R':
                    node_colors.append('red')
                else:
                    node_colors.append(node_color)
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)
            #print(G.nodes)

            # draw edges
            nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
            nx.draw_networkx_edges(
                G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
            )

            # node labels
            nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
            # edge weight labels whith prescision
            edge_labels = dict([((u, v,), f"{d['weight']:.2f}") for u, v, d in G.edges(data=True)])
            nx.draw_networkx_edge_labels(G, pos, edge_labels)

            ax = plt.gca()
            ax.margins(0.08)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        def draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8,
                 edge_color='gray'):

            # Определяем цвет вершин в зависимости от их типа.
            node_colors = []
            for node in G.nodes:
                if G.nodes[node]['object'] == 'R':
                    node_colors.append('red')
                else:
                    node_colors.append(node_color)

            # Рисуем граф.
            nx.draw(G, pos, with_labels=with_labels, font_weight=font_weight, node_size=node_size,
                    node_color=node_colors, font_size=font_size, edge_color=edge_color)

        def drawClusters(self, groups):
            Graph = self.G
            pos = dict(enumerate(self.sample, 0))
            g = Clustering.graphDivision(Clustering(Graph), groups)

            # draw nodes
            node_color = 'skyblue'
            node_colors = []
            for i in range(0, len(g)):
                # Рисуем граф кластеров
                DrawGraph.draw(g[i], pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue',
                               font_size=8, edge_color='gray')

            ax = plt.gca()
            ax.margins(0.08)
            plt.axis("off")
            #plt.tight_layout()
            plt.show()

        def drawCmap(clusters, sample):
            plt.figure(figsize=(8, 6))
            pos = dict(enumerate(sample, 0))
            for i in range(0, len(clusters)):
                # Получаем значения charge для всех узлов
                charges = nx.get_node_attributes(clusters[i], 'charge')
                charge_values = list(charges.values())

                # Нормируем значения charge для отображения цветов
                norm = mcolors.Normalize(vmin=min(charge_values), vmax=max(charge_values))
                cmap = plt.get_cmap('Reds')

                # Определяем цвета для узлов на основе значений charge
                node_colors = [cmap(1 - norm(charges[node])) for node in clusters[i].nodes]


                nx.draw(clusters[i], pos, with_labels=True, node_color=node_colors, node_size=500, edge_color='gray',
                        font_weight='bold')

                # Отображаем цветовую шкалу
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                #plt.colorbar(sm, label='сharge')

            plt.show()


class ProbabilityDependencies():
    def dependencyGraphs():
        # Создание массивов значений для r и f
        r_values = np.arange(100, 2001, 200)  # от 100 до 2000 с шагом 200
        f_values = np.arange(20, 101, 10)  # от 20 до 100 с шагом 10

        # Построение графиков для p1 и p2 по r при изменяющемся f
        plt.figure(figsize=(12, 6))
        for f in f_values:
            p1_values = [DeterminingTheEdgeWeight.p1(r, f) for r in r_values]
            p2_values = [DeterminingTheEdgeWeight.p2(r, f) for r in r_values]
            plt.plot(r_values, p1_values, label=f'p1, f={f} kHz')
            plt.plot(r_values, p2_values, label=f'p2, f={f} kHz', linestyle='dashed')
        plt.xlabel('Расстояние между сенсорами, м')
        plt.ylabel('Вероятность')
        plt.title('Графики зависимости вероятности от расстояния между сенсорами при разных частотах')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Построение графиков для p1 и p2 по f при изменяющемся r
        plt.figure(figsize=(12, 6))
        for r in r_values:
            p1_values = [DeterminingTheEdgeWeight.p1(r, f) for f in f_values]
            p2_values = [DeterminingTheEdgeWeight.p2(r, f) for f in f_values]
            plt.plot(f_values, p1_values, label=f'p1, r={r} m')
            plt.plot(f_values, p2_values, label=f'p2, r={r} m', linestyle='dashed')
        plt.xlabel('Частота передачи данных, кГц')
        plt.ylabel('Вероятность')
        plt.title('Графики зависимости вероятности от частоты передачи данных при разных расстояниях')
        plt.legend()
        plt.grid(True)
        plt.show()


sample = PointGenerator.gen(PointGenerator(10, 10, 0.3, 3))
graph = GraphGenerator.abc(GraphGenerator(sample, 5, 300))
groups = Clustering.createGroups(Clustering(graph))
print(groups)
clusters = Clustering.graphDivision(Clustering(graph), groups)

for i in range(100): # Количество передач данных
    TransferData.transfer(TransferData(clusters, groups))

for i in range(len(groups)):
    print(groups[i][-1], clusters[i].nodes[groups[i][-1]])
DrawGraph.drawTheEntireGraph(DrawGraph(graph, sample))
DrawGraph.drawClusters(DrawGraph(graph, sample), groups)
DrawGraph.drawCmap(clusters, sample)

ProbabilityDependencies.dependencyGraphs()