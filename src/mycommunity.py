from __future__ import division, print_function
import networkx as nx
import sys
import community as community
from pprint import pprint

"""
Undirected graph
no edge weight
"""

class Community(object):
	"""docstring for Community."""
	def __init__(self, graph, partition=None):
		super(Community, self).__init__()
		self._graph = graph
		self._nodes = list(graph.nodes())
		self._adj = nx.to_dict_of_lists(self._graph)
		self._adj = {n: set(self._adj[n]) for n in self._adj}
		self._nodes_count = self._graph.number_of_nodes()
		self._edges_count = self._graph.number_of_edges()
		self._degree = nx.degree(self._graph)
		if partition is None:
			self._dendrogram = {self._nodes[i]:[i] for i in xrange(0, len(self._nodes))}
		else:
			self._dendrogram = {u:[partition[u]] for u in partition}
		self._level = 0
		self._com_index = len(self._nodes)  # index used for naming communities
		self._igraph = self.initInducedGraph()
		self.__PASS_MAX = 100

		self._tree = None


	def initInducedGraph(self):
		graph = nx.Graph()
		nodes = [self._dendrogram[v][0] for v in self._dendrogram]
		edges = [(self._dendrogram[e[0]][0], self._dendrogram[e[1]][0]) for e in self._graph.edges()]
		graph.add_nodes_from(nodes)
		graph.add_edges_from(edges)
		return graph


	def updateInducedGraph(self, i, j):
		"""
		Update induced graph by merging i and j nodes
		The name of the new node is k
		"""
		self._igraph = nx.contracted_nodes(self._igraph, i, j)
		self._igraph = nx.relabel_nodes(self._igraph, {i: self._com_index, j: self._com_index}, copy=True)

		# Remove self loops
		snodes = self._igraph.nodes_with_selfloops()
		sedges = [(n,n) for n in snodes]
		self._igraph.remove_edges_from(sedges)
		#print('induced', i, j, self._igraph.number_of_nodes(), self._com_index)


	def partitionToCommunity(self, partition):
		com = {}
		for n in partition:
			if partition[n] not in com:
				com[partition[n]] = set()
			com[partition[n]].update([n])
		return com


	def communityToPartition(self, community):
		partition = {}
		for c in community:
			for u in community[c]:
				partition[u] = c
		return partition


	def dendrogramToCommunity(self, level=None):
		if level is None:
			level = self._level

		com = {}
		for u in self._dendrogram:
			c = self._dendrogram[u][level]
			if c not in com:
				com[c] = set()
			com[c].update([u])

		return com

	
	def updateDendrogram(self, community):
		for c in community:
			for u in community[c]:
				if c not in self._dendrogram[u]:
					self._dendrogram[u].append(c)
		self._level += 1


	def mergeCommunities(self, community, c1, c2):
		"""
		Merge communities c1 and c2, and return the resulting communities
		"""
		self._com_index += 1
		#print('merge', c1, c2, self._com_index)
		com = {c:community[c] for c in community if c != c1 and c != c2}
		com[self._com_index] = community[c1].union(community[c2])
		return com


	def modularity(self, com):
		partition = self.communityToPartition(com)
		#return community.modularity(partition, self._graph)

		q = 0
		if self._edges_count == 0:
			return -1

		for u in self._nodes:
			q1 = 0
			q2 = 0
			# Neighbors in the same community
			n1 = self._adj[u].intersection(com[partition[u]])
			# Nodes in same community but not neighbors
			n2 = com[partition[u]].difference(self._adj[u])

			for v in n1:
				q1 += self._degree[v]
			q += len(n1) - self._degree[u]*q1/(2*self._edges_count)
	
			for v in n2:
				q2 += self._degree[v]
			q -= self._degree[u]*q2/(2*self._edges_count)
	
		return q/(2*self._edges_count)


	def generateDendrogram(self, stop_max=True):
		community = self.dendrogramToCommunity()
		tpass = 0
		while True:
			merged = False
			cindex = community.keys()

			max_modularity = self.modularity(community)
			max_community = None

			removed = []
			for c1 in cindex:
				if c1 in removed:
					#print(removed)
					continue
				mc = None
				mods = {}
				#print(self._igraph.neighbors(c1))
				if len(self._igraph[c1]) > 0:
					candidates = self._igraph.neighbors(c1)
				else:
					candidates = [n for n in self._igraph.nodes() if n != c1]

				for c2 in candidates:
					com = self.mergeCommunities(community, c1, c2)
					mod = self.modularity(com)
					mods[c2] = mod
					if mod >= max_modularity:
						max_community = com
						max_modularity = mod
						mc = c2

				if mc is not None:
					#print(len(community), c1, mc, max_modularity)
					removed += [c1, mc]
					community = self.mergeCommunities(community, c1, mc)
					self.updateDendrogram(community)
					self.updateInducedGraph(c1, mc)
					merged = True
					tpass = 0
				elif not stop_max and len(mods) > 0:
					mc = max(mods, key=mods.get)
					removed += [c1, mc]
					community = self.mergeCommunities(community, c1, mc)
					self.updateDendrogram(community)
					self.updateInducedGraph(c1, mc)
					merged = True
					tpass = 0
					#print(len(community),c1, mc, mods[mc])

			if not merged:
				break
			if not stop_max and len(community) <= 1:
				break
			tpass += 1
			if tpass >= self.__PASS_MAX:
				break
		return self._dendrogram

	def communityAtLevel(self, l=0):
		coms = set([self._dendrogram[u][l] for u in self._nodes])
		community = {c:[] for c in coms}

		for u in self._nodes:
			community[self._dendrogram[u][l]].append(u)
		return community

	def flattenDendrogram(self, tree, parent=None, flat=None):
		if flat is None:
			flat = {}
		keys = tree.keys()
		for u in keys:
			flat[u] = {'parent': parent, 'siblings': [x for x in keys if x != u], 'children': tree[u].keys()}
			flat = self.flattenDendrogram(tree=tree[u], parent=u, flat=flat)

		return flat

	def nodesInCommunity(self, com):
		nodes = []
		for u in self._nodes:
			if com in self._dendrogram[u]:
				nodes.append(u)
		return nodes

	def numberOfLevels(self):
		return self._level

	def communityTree(self):
		"""
		Returns the tree of communities
		"""
		#print(self._nodes[0])
		#print(self._dendrogram.keys())
		root = self._dendrogram[self._nodes[0]][-1]
		tree = {}
		for u in self._nodes:
			ttree = tree
			dend = []
			for d in self._dendrogram[u]:
				if d not in dend:
					dend.append(d)
			dend.reverse()

			for c in dend:
				if c in ttree:
					ttree = ttree[c]
				else:
					ttree[c] = {}
					ttree = ttree[c]

		self._tree = tree
		return self._tree

	def communityIds(self):
		"""
		Returns the ids of all the communities at all levels
		"""
		if self._tree is None:
			self.communityTree()
		self._com_names = self._getTreeNodes(self._tree)
		return self._com_names

	def _getTreeNodes(self, data):
		"""
		Returns the ids of the communities
		"""
		if len(data) == 0:
			return []
		else:
			nodes = []
			for n in data:
				nodes.append(n)
				nodes += self._getTreeNodes(data[n])
			return nodes

	def getRoot(self):
		"""
		The last community id is always the root
		"""
		return self._dendrogram[self._nodes[0]][-1]



if __name__ == '__main__':
	g = nx.Graph()
	g.add_edge('a', 'b')
	g.add_edge('b', 'c')
	g.add_edge('a', 'c')
	g.add_edge('b', 'd')
	g.add_edge('d', 'e')
	g.add_edge('d', 'f')
	g.add_edge('e', 'f')

	community = {0: set(['a', 'b', 'c']), 1:set(['d', 'e', 'f'])}
	com = Community(g)
	print(com.modularity(community))


	"""
	fname = sys.argv[1]

	graph = nx.read_edgelist(fname, delimiter=',')
	#graph.add_node('test')
	#graph = max(nx.connected_component_subgraphs(graph), key=len)
	#graph = nx.fast_gnp_random_graph(50, 0.1)
	print(nx.info(graph))

	partition = community.best_partition(graph)
	pprint(community.modularity(partition, graph))

	com = Community(graph, partition=partition)
	#com = Community(graph, partition=None)
	com.generateDendrogram(stop_max=False)
	pprint(com.numberOfLevels())
	tree = com.communityTree()
	#pprint(tree)
	den = com.flattenDendrogram(tree)
	#pprint(den)
	#print(com.communityAtLevel(5))
	#print(com.modularity(partition))

	#pprint(com.communityIds())
	"""
