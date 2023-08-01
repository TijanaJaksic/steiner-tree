import sys
import numpy as np

import matplotlib.pyplot as plt

import rustworkx as rx
from rustworkx.visualization import mpl_draw

#metoda koja proverava da li je stablo povezano
#da li je drvo
#da li ima sve terminale

#GRASP
#fix set search
#ucitavanje + poredjenje sa tudjim rezultatima: ant colony, grasp w path relinking, ...

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.num_edges = 0
        self.terminals = set() #mzd bolje lista. za sad ostajem sa skupom
        self.steiner_tree = Steiner()
        #pravim listu pripadnosti cvorova stajnerovom stablu
        # aff_list[k] == True AKKO  k-ti cvor pripada Steinerovom stablu
        #inicijalizujemo je kao svuda False
        self.steiner_affiliation_list = [False]*num_vertices#__________________________________________________________ne koristim

        self.min_paths_found = False #ne nalazim najkrace puteve pri inicijalizaciji
        self.steiner_tree_found = False #nije mi nadjeno stajnerovo stablo pri inicijalizacije

        #adjacency list:
        self.adj_list = {}
        for i in range(num_vertices):
            self.adj_list[i] = []
        
        #adjacency matrix:
        self.adj_matrix = np.ones((num_vertices, num_vertices))
        self.adj_matrix.fill(float('inf'))

        #rustworkx graph:
        self.graph = rx.PyGraph()
        self.graph.add_nodes_from(range(num_vertices))    

    def add_edge(self, v1, v2, w):
        self.min_paths_found = False #ako smo nasli puteve pa menjali grane, kao da nismo nasli puteve
        self.steiner_tree_found = False #ako smo dodali granu mozda nam vise ne valja stein stablo

        self.num_edges += 1
        if v1 < 0 or v1 >= self.num_vertices or v2 < 0 or v2 >= self.num_vertices:
            print("Vertice indexes out of range.")
            sys.exit(1)

        #adjacency list:
        self.adj_list[v1].append((v2, w))
        self.adj_list[v2].append((v1, w))

        #adjacency matrix:
        self.adj_matrix[v1][v2] = w
        self.adj_matrix[v2][v1] = w

        #rustworkx graph:
        self.graph.add_edge(v1, v2, w)

    def add_terminal(self, v):
        self.steiner_tree_found = False #ako smo dodali terminal, staro steiner stablo nam mozda vise ne valja

        if v < 0 or v >= self.num_vertices:
            print("Vertice indexes out of range.")
            sys.exit(1)
        self.terminals.add(v)
    
    def __min_distances_help(self):
        if self.min_paths_found == True: #ako su vec nadjeni necemo ponovo da ih racunamo
            return
        
        self.min_distance_matrix = np.ones((self.num_vertices, self.num_vertices))

        self.min_path_matrix = np.zeros((self.num_vertices, self.num_vertices), dtype=int)

        for i in range(0, self.num_vertices):
            for j in range(0, self.num_vertices):
                self.min_distance_matrix[i][j] = self.adj_matrix[i][j]
                if self.min_distance_matrix[i][j] != float('inf') and i!=j:
                    self.min_path_matrix[i][j] = i
                else:
                    self.min_path_matrix[i][j] = int(-1)
        
        for k in range(0, self.num_vertices):
            for i in range(0, self.num_vertices):
                for j in range(0, self.num_vertices):
                    if self.min_distance_matrix[i][k] == float('inf') or self.min_distance_matrix[k][j] == float('inf') or i == j:
                        continue
                    if self.min_distance_matrix[i][j] > self.min_distance_matrix[i][k] + self.min_distance_matrix[k][j]:
                        self.min_distance_matrix[i][j] = self.min_distance_matrix[i][k] + self.min_distance_matrix[k][j]
                        self.min_path_matrix[i][j] = self.min_path_matrix[k][j]

        self.min_paths_found = True #nasli smo min puteve i to pamtimo
    #vraca dobro
    def min_distances(self):
        self.__min_distances_help()
        for i in range(0, self.num_vertices):
            for j in range(0, self.num_vertices):
                print(self.min_distance_matrix[i][j], end = " ")
            print(" ")
        return self.min_distance_matrix
    #vraca dobro
    def min_paths(self):
        self.__min_distances_help()
        for i in range(0, self.num_vertices):
            for j in range(0, self.num_vertices):
                print(self.min_path_matrix[i][j], end = " ")
            print(" ")
        return self.min_path_matrix
    
    def find_steiner_tree(self):
        if self.steiner_tree_found == True:
            return
        self.__min_distances_help()

        #kopiram skup terminala jer mi treba skup koji cu usput da praznim
        #skup pamti T\S ... (T - skup terminala; S - skup cvorova u Stajnerovom stablu)
        terminals_set = set()
        for t in self.terminals:
            terminals_set.add(t)

        t = terminals_set.pop()
        self.steiner_tree.add_vertice(t)
        
        while(terminals_set): #u pythonu prazan skup nosi vrednost False!
            (t, s) = self.__find_vertice_closest_to_stein_tree(terminals_set)
            terminals_set.remove(t)
            self.add_path_to_stein_tree(s, t)

    #dodajemo najkraci put izm v1 i v2 u stajnerovo stablo
    def add_path_to_stein_tree(self, v1, v2):#################################################################################
        if v1 == v2:
            return
        #treba mi matrica najkracih puteva i matrica povezanosti
        self.__min_distances_help()

        s = self.min_path_matrix[v1][v2] #s stands for "srednji cvor" - cvor koji prethodi cvoru v2 u najkracem putu v1->v2

        while s != v1:
            self.steiner_tree.add_edge(s, v2, self.adj_matrix[s][v2])
            self.steiner_affiliation_list[s] = True#____________________________________________________________________ne koristim
            self.steiner_affiliation_list[v2] = True#___________________________________________________________________ne koristim
            v2 = s
            s = self.min_path_matrix[v1][v2]
        self.steiner_tree.add_edge(v1, v2, self.adj_matrix[v1][v2])
    
    #za proslednjen cvor v nalazi najblizi cvor iz stajnerovog stabla i udaljenost
    def __find_distance_to_stein_tree(self, v):
        closest_verice = -1
        distance = float("inf")
        for s in self.steiner_tree.vertices:
            if distance > self.min_distance_matrix[v][s]:
                distance = self.min_distance_matrix[v][s]
                closest_verice = s
        return (closest_verice, distance)
    
    def __find_vertice_closest_to_stein_tree(self, vertice_set):
        closest_vertice_from_given_set = -1
        closest_vertice_steiner = -1
        distance = float('inf')
        for v in vertice_set:
            (s, d) = self.__find_distance_to_stein_tree(v)
            if distance > d:
                distance = d
                closest_vertice_from_given_set = v
                closest_vertice_steiner = s
        return (closest_vertice_from_given_set, closest_vertice_steiner)
    
    def print(self):
        for vertice in range(0, self.num_vertices):
            print(vertice, end = ": ")
            for neighbour in self.adj_list[vertice]:
                print(neighbour, end = " ")
    
class Steiner:
    def __init__(self):
        self.num_vertices = 0
        self.num_edges = 0

        #nisam dodala matricu povezanosti jer drvo pravim usput - dodajem cvor po cvor, 
        # previse cesto bih morala da menjam dimenziju matrice
        self.adj_list = {}
        self.vertices = set()
        self.edges = [] #pamti samo parove cvorova povezane granom ne i njihovu tezinu

    def add_vertice(self, num):
        if num in self.vertices:
            return
        self.vertices.add(num)
        self.adj_list[num] = []

    def add_edge(self, v1, v2, w):
        if not v1 in self.vertices:
            self.add_vertice(v1)
        if not v2 in self.vertices:
            self.add_vertice(v2)
        #nema smisla posmatrati grafove sa vise grana izmedju istog para cvorova jer trazimo STAJNEROVA stabla
        #zato proverim samo da li postoji vec grana izmedju v1 i v2, ne i da li je to moja grana tezine w
        #znaci ako se grana nalazi u stablu, ne radim nista
        #DA LI IMA VISE SMISLA DA PAROVE GRANA PAMTIM DRUGACIJE?
        if v1 < v2:
            vv1 = v1
            vv2 = v2
        elif v1 > v2:
            vv1 = v2
            vv2 = v1
        else:
            return #jer nema smisla da dodamo granu izmedju v i v
        if (vv1, vv2) in self.edges:
            return
        self.adj_list[v1].append((v2, w))
        self.adj_list[v2].append((v1, w))
        self.edges.append((vv1, vv2))

    def print(self):
        for vertice in self.vertices:
            print(vertice, end = ": ")
            for neighbour in self.adj_list[vertice]:
                print(neighbour, end = " ")

#ucitavamo graf iz fajla
with open('test.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        x = line.split()
        if x == []:
            continue
        if x[0] == 'Nodes':
            g = Graph(int(x[1]))
        if x[0] == 'Edges':
            num_edges = x[1]
        if x[0] == 'E':
            g.add_edge(int(x[1])-1, int(x[2])-1, int(x[3]))
        if x[0] == 'Terminals':
            num_terminals = int(x[1])
        if x[0] == 'T':
            g.add_terminal(int(x[1])-1)

    g.find_steiner_tree()
    g.steiner_tree.print()
    print(g.terminals)

    #g.print()
    #g.min_paths()
    #g.min_distances()

    #g.add_path_to_stein_tree(0, 5)
    #g.steiner_tree.print()
    
#nacrtaj ucitano:
#mpl_draw(g.graph)

#plt.show()