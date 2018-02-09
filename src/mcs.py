"""
Script for MCS
"""

from __future__ import division, print_function

import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from my_community import mycommunity

import networkx as nx
import community
import csv
import numpy as np
import random
import cPickle as pickle
import operator
from pprint import pprint
import operator
import traceback
from sklearn.metrics.cluster import normalized_mutual_info_score



class MultiArmBandit(object):
	"""docstring for MultiArmBandit."""
	def __init__(self, arms, est = None, epsilon=0.1):
		super(MultiArmBandit, self).__init__()
		if len(set(arms)) != len(arms):
			print('error')
			return False
		self.arms = arms
		if est is None or len(est) == 0:
			self.est = np.ones(len(self.arms))
			self.est_hist = [[] for _ in xrange(0,len(self.arms))]
		else:
			self.est = np.array(est)
			self.est_hist = [[0] for e in est]
		self.epsilon = epsilon
		

	def nextArm(self, arms=None, rprob=0.0):
		if len(self.arms) == len(self.removed):
			return False
		return self.epsilonGreedy(arms=arms, rprob=rprob)

	def getEst(self, arm):
		return self.est[arm]

		
	def epsilonGreedy(self, arms=None, rprob=0.0):
		"""
		With probability epsilon, select the best arm.
		With probability (1-epsilon), select a random arm.
		"""
		if arms is None or len(arms) == 0:
			arms = self.arms
		
		r = np.random.random()
		if r > self.epsilon:
			# Select the best arm
			# If multiple best arms, select randomly
			est = np.array([self.est[i] if self.arms[i] in arms\
			 else np.NINF for i in xrange(0, len(self.arms))])	
			if np.random.random() > rprob:
				# Select highest
				i = np.random.choice(np.where(est == est.max())[0])
			else:
				# Select lowest
				i = np.random.choice(np.where(est == est.min())[0])

			return self.arms[i]
		
		# Otherwise, return random arm
		i = np.random.choice(range(0, len(arms)))
		return arms[i]


	def updateEst(self, reward, arm):
		i = self.arms.index(arm)
		self.est_hist[i].append(reward)
		self.est[i] = np.mean(self.est_hist[i][-self._window_size :])
		return self.est[i]



class EdgeOverlap(object):
	"""Class for handling layer importance realated methods"""
	def __init__(self, layer_count):
		super(EdgeOverlap, self).__init__()
		# Initialize with some default values
		self._importance = [[10.0] for _ in xrange(0,layer_count)]


	def _updateBaseGraph(self, graph, nodes=None):
		"""
		The graph against which importance will be calculated
		"""
		self._base_graph = graph 							
		
		if nodes is not None:
			self._base_graph = self._base_graph.subgraph(nodes)

		self._base_nodes = set(list(self._base_graph.nodes()))
		self._base_edges = set([frozenset(e) for e in self._base_graph.edges()])


	def _edgeOverlap(self, graph):
		"""
		Fraction of edges in graph that are also in base graph
		If nodes is None, all the nodes in graph are considered.
		Otherwise only subgraph containing nodes is considered.
		"""
		sg = graph.subgraph(self._base_nodes)

		if sg.number_of_edges() == 0:
			# If there are no edges in subgraph, return False
			return False

		edges = set([frozenset(e) for e in sg.edges()])
		
		return len(self._base_edges.intersection(edges)) / len(edges)


	def updateEdgeOverlap(self, graphs, nodes=None):
		"""
		Update the importance of all layers in graphs, and nodes
		"""
		self._updateBaseGraph(graphs[0], nodes)

		for i in xrange(0, len(graphs)):
			importance = self._edgeOverlap(graphs[i])
			if importance is not False:
				self._importance[i].append(importance)


	def getEdgeOverlap(self, layer=None):
		if layer is not None:
			return self._importance[layer][-1]
		else:
			return [i[-1] for i in self._importance]
					


class Budget(object):
	"""
	Class for the budget allocations
	"""
	def __init__(self, max_budget, layer_costs):
		super(Budget, self).__init__()
		self._budget_max = max_budget
		self._budget_left = max_budget
		self._budget_consumed = 0
		self._layer_costs = layer_costs

		# Initial number of slices
		self._slices = 20

		# The budget consumed when slice was last updated
		self._slices_last_update = 0 	


	def initializeBudget(self):
		"""
		Allocate enough budget such that same number of queries can 
		be made in each layer
		"""
		budget = self._budget_left/self._slices
		total_cost = sum(self._layer_costs)
		allocated = []

		for c in self._layer_costs:
			allocated.append(budget * c / total_cost)

		return allocated


	def consumeBudget(self, cost):
		self._budget_consumed += cost
		self._budget_left -= cost


	def updateSlices(self):
		"""
		Update number of slices based on cost consumed since last update
		"""
		if self._budget_consumed == self._slices_last_update:
			return True
		cost = self._budget_consumed - self._slices_last_update
		self._slices = np.ceil(self._budget_left / cost)
		#print(cost, self._budget_left, self._slices)
		self._slices_last_update = self._budget_consumed


	def allocateBudget(self, weights):
		"""
		Allocate the budget based on weights
		Layers with high weight gets more budget
		Budget for layer 0 depends only on layer cost
		"""
		if len(weights) != len(self._layer_costs):
			return False

		budget = self._budget_left/self._slices
		allocation = []

		# Budget for layer 0 
		b0 = self._layer_costs[0] * budget / np.sum(self._layer_costs)
		allocation.append(b0)

		# Remainig budget
		budget -= b0

		# Total weights excluding layer 0
		total_weight = np.sum(weights[1:])

		for w in weights[1:]:
			b = budget * w / total_weight
			allocation.append(b)

		return allocation


	def getBudgetLeft(self):
		return self._budget_left

	def getBudgetConsumed(self):
		return self._budget_consumed
		


class Evaluation(object):
	"""
	Class for evaluation
	"""
	def __init__(self, graphs):
		super(Evaluation, self).__init__()
		
	
	def _getPartition(self, graph):
		return community.best_partition(graph, randomize=False)


	def _communityToPartition(self, com):
		part = {}
		for c in com : part.update(dict.fromkeys(com[c], c))
		return part
	

	def _getCommunity(self, part):
		com = {}
		for c in set(part.values()):
			com[c] = [u for u, k in part.items() if k == c]

		# Make sure we do not consider the singleton nodes
		com = {c:n for c, n in com.items() if len(n) > 1}

		return com


	def partitionDistance(self, part1, part2, nodes):
		"""
		Compute the partiton distance between communities c1 and c2
		"""
		c1 = self._getCommunity(part1)
		c2 = self._getCommunity(part2)

		c1 = {c:k.intersection(nodes) for c, k in c1.items()}
		c2 = {c:k.intersection(nodes) for c, k in c2.items()}

		m = xrange(0, max(len(c1), len(c2)))

		m0 = dict.fromkeys(c2.keys(), 0)
		mat = dict.fromkeys(c1.keys(), dict(mat0))
		#mat = {i: {j: 0 for j in c2} for i in c1}

		total = 0
		for i, k0 in c1.items():
			for j, k1 in c2.items():
				mat[i][j] = len(k0[i].intersection(k1[j]))
				total += mat[i][j]

		if total <= 1:
			return 1.0

		assignment = []
		rows = c1.keys()
		cols = c2.keys()

		while len(rows) > 0 and len(cols) > 0:
			mval = 0
			r = -1
			c = -1
			for i in rows:
				for j in cols:
					if mat[i][j] >= mval:
						mval = mat[i][j]
						r = i 
						c = j
			rows.remove(r)
			cols.remove(c)
			assignment.append(mval)
		
		dist = total - np.sum(assignment)

		if np.isnan(dist/total):
			return 0
			
		return dist/total


		
class RNDSample(object):
	"""docstring for RNDSample"""
	def __init__(self, graph, sample, lcosts, queried, budget):
		super(RNDSample, self).__init__()
		self._sample = sample
		self._graph = graph
		self._lcosts = lcosts
		
		self._queried = queried
		self._unqueried = [set([]) for _ in self._sample]

		# reset prob for random walk
		self._alpha = 0.1 									
		self._budget = budget

		self._initializeSample()


	def _initializeSample(self):
		"""
		Initialize sample by adding some random nodes to samples
		"""
		for i in xrange(1, len(self._sample)):
			# Initial nodes to start
			nodes = np.random.choice(list(self._graph[i].nodes()), 10,\
			 replace=False)
			self._sample[i].add_nodes_from(nodes)
			self._unqueried[i].update(nodes)


	def sample(self, budget):
		"""
		Sample graph with random walk
		"""
		for i in xrange(1, len(self._sample)):
			sample = self._sample[i]
			graph = self._graph[i]
			queried = self._queried[i]
			unqueried = self._unqueried[i]
			
			if len(unqueried) > 0:
				# Start with a random unqueried node
				u = np.random.choice(list(unqueried))
			else:
				# If no queried node, start with random unqueried node
				# in random layer
				l = np.random.choice(range(0, len(self._unqueried)))
				if len(self._unqueried[l]) > 0:
					u = np.random.choice(list(self._unqueried[l]))
				else:
					u = None
			c = 0

			while c < budget[i] and u is not None\
			 and self._budget.getBudgetLeft() > 0:
				c += self._layer_costs[i]
				self._budget.consumeBudget(self._layer_costs[i])

				neighbors = set(list(graph.neighbors(u)))
				edges = [(u,v) for v in neighbors]
				sample.add_edges_from(edges)

				queried.update([u])
				unqueried.update(neighbors)
				unqueried.difference_update(queried)
				
				# If no unqueried node, stop
				if len(unqueried) == 0:
					break

				candidates = set(neighbors).difference(queried)
				
				if np.random.random_sample() > self._alpha\
				 and len(candidates) > 0:
					u = np.random.choice(list(candidates))
				elif len(unqueried) > 0:
					u = np.random.choice(list(unqueried))
				else:
					break



class CommunityManager(object):
	"""docstring for CBanditManager"""
	def __init__(self, hcommunity):
		super(CommunityManager, self).__init__()
		self._hcommunity = hcommunity
		self._initalCommunities()
		self._generateMapping()


	def _getComName(self, layer, i):
		"""
		Return com name given layer and id
		"""
		return self._map[layer][i]


	def _initalCommunities(self):
		"""
		The two initial communities for all layers
		"""
		roots = self._hcommunity.getRootCommunity()
		self._active_communities = []
		self._rewards = []
		self._crewards = []
		
		for l in xrange(0, self._hcommunity.getLayerCount()):
			coms = self._hcommunity.getChildren(l, roots[l])
			self._active_communities.append(coms)
			self._crewards.append({c:[] for c in coms})


	def getActiveCommunities(self, layer):
		return self._active_communities[layer]


	def updateCReward(self, layer, cid, value):
		#cid = self._map[layer][cid]
		self._rewards.append(value)
		self._crewards[layer][cid].append(value)


	def switchArm(self, layer):
		"""
		Check rewards to check if active community need to be changed
		"""
		if np.any([len(r) for l, r in self._crewards[layer].items()] < 5) :
			return False

		rewards = self._crewards[layer]
		cid = self._active_communities[layer]

		aval = np.mean(self._rewards)
		astd = np.std(self._rewards)

		mval = {c:np.mean(rewards[c]) for c in cid}
		sval = {c:np.std(rewards[c]) for c in cid}

		changed = False

		# If both arms have very low rewards, swith up
		if np.all([mval[c] + sval[c] for c in cid] < aval):
			self.switchArmUp(layer, cid[0])
			changed = True
		elif mval[cid[0]] < mval[cid[1]] - sval[cid[1]]:
			# If arm 0 is very much lower than 1, swith down to 1
			self.switchArmDown(layer, cid[1])
			changed = True
		elif mval[cid[1]] < mval[cid[0]] - sval[cid[0]]:
			self.switchArmDown(layer, cdi[0])
			changed = True

		if changed:
			cid = self._active_communities[layer]
			self._crewards[layer] = {c:[] for c in cid}


	def switchArmDown(self, layer, cid):
		"""
		Switch to a lower level of community from comid
		"""
		active = self.getActiveCommunities(layer)

		if comid not in active:
			return None

		if self._hcommunity.checkLeaf(layer, cid):
			# If leaf, return cid and sibling
			return (cid, self._hcommunity.getSibling(layer, cid)[0])

		return self._hcommunity.getChildren(layer, cid)


	def switchArmUp(self, layer, cid):
		active = self.getActiveCommunities(layer)

		if comid not in active:
			return None

		parent = self._hcommunity.getParent(layer, cid)
		
		if self._hcommunity.checkLeaf(layer, parent):
			# if parent is root, return self and sibling
			return (cid, self._hcommunity.getSibling(layer, cid)[0])

		return (parent, self._hcommunity.getSibling(layer, parent)[0])



class BanditManager(object):
	"""Manages the multiple bandits"""
	def __init__(self, graph, sample, queried):
		super(BanditManager, self).__init__()
		self._epsilon = 0.2
		
		self._graph = graph
		self._sample = sample
		self._queried = queried
		self._layers = range(0, len(graph))

		self._lbandit = MultiArmBandit(self._layers, epsilon=self._epsilon)
		self._cbandit = [None for _ in self._layers]
		self._rbandit = [None for _ in self._layers]

		#self.initializeBudget()


	def initializeRBandits(self):
		self._rolmanager = RoleManager(self._sample, self._queried)
		self._rbandit = [MultiArmBandit(range(0, len(self._rolmanager.getRoles())),\
		 epsilon=self._epsilon) for _ in self._layers]


	def initializeCBandits(self):
		self._hcommunity = CommunitHeirarchy(self._sample)
		self._commanager = CommunityManager(self._hcommunity)

		self._cbandit = [MultiArmBandit(self._hcommunity.getCommunityIds(l).values(),\
		 epsilon=self._epsilon) for l in self._layers]


	def _nextArms(self):
		"""
		Get the next layer and role
		"""
		larm, _ = self._lbandit.nextArm()
		carm, _ = self._cbandit[larm].nextArm(arms=self._commanager.getActiveCommunities(larm))
		rarm, _ = self._rbandit[larm].nextArm()

		self._arm = (larm, carm, rarm)


	def getArms(self):
		return self._arm


	def updateReward(self, reward):
		self._lbandit.updateEst(reward[0], self._arm[0])
		self._rbandit[self._arm[0]].updateEst(reward[2], self._arm[2])
		self._updateRewardCBandit(reward[1], True)
		self._commanager.updateCReward(self._arm[0], self._arm[1], reward[1])
		self._commanager.switchArm(self._arm[0])

	def _updateRewardCBandit(self, reward, updateparent=False):
		"""
		If updateparent is True, update the parents of community in heirarchy too
		"""
		cids = [self._arm[1]]

		if updateparent:
			cids += self._hcommunity.getAncestors(self._arm[0], self._arm[1])

		for cid in cids:
			self._cbandit[self._arm[0]].updateEst(reward, cid)


	def getNode(self):
		self._nextArms()
		
		candidates = set(list(self._hcommunity.getNodes(self._arm[0], self._arm[1])))
		candidates.difference_update(self._queried[0])

		return self._rolmanager.getNode(self._arm[0], self._arm[2], candidates)



class RoleManager(object):
	"""docstring for RoleManager"""
	def __init__(self, sample, queried):
		super(RoleManager, self).__init__()
		self._sample = sample
		self._queried = queried

		self._roles = [('degree', 'highest'), ('degree', 'lowest'), \
		 ('betweeness', 'highest'), ('betweeness', 'lowest'),
		 ('clustering', 'highest'), ('clustering', 'lowest')]
		

	def getRoles(self):
		return self._roles


	def getNode(self, layer, role, nodes=None):
		"""
		Get node satisfying role from layer

		If nodes is given, select only from that list
		"""
		s = self._sample[layer]
		r = self._roles[role]

		candidates = set(list(s.nodes()))
		candidates.difference_update(self._queried[0])

		if nodes is not None:
			candidates.intersection(nodes)

		# If no candidiates, return false
		if len(candidates) == 0:
			return False

		# Get nodes and values
		if r[0] == 'degree':
			vals = nx.degree_centrality(s)
		elif r[0] == 'betweeness':
			vals = nx.betweenness_centrality(s, k=10)
		elif r[0] == 'closeness':
			vals = nx.closeness_centrality(s)
		elif r[0] == 'clustering':
			vals = nx.clustering(s)

		# Filter to only nodes in canditate list
		candidates = {u:vals[u] for u in candidates}

		# Sort by the value
		candidates = sorted(candidates, key=candidates.get)

		#print(r, vals[candidates[0]], vals[candidates[-1]])
		# Return highest or lowest depending on role
		if r[1] == 'lowest':
			return candidates[0]
		else:
			return candidates[-1]
	


class CommunitHeirarchy(object):
	"""docstring for CommunitHeirarchy"""
	def __init__(self, sample):
		super(CommunitHeirarchy, self).__init__()
		self._sample = sample
		self._initializeCommunities()


	def getLayerCount(self):
		return len(self._sample)


	def _initializeCommunities(self):
		"""
		Intitialize communites for each of the layers
		"""
		self._ocom = []
		self._dendrogram = []
		self._com_ids = []

		for i in xrange(0, len(self._sample)):
			partition = community.best_partition(self._sample[i], randomize=False)
			com = mycommunity.Community(self._sample[i], partition)
			com.generateDendrogram(stop_max=False)
			tree = com.communityTree()
			ids = com.communityIds()

			self._ocom.append(com)
			self._dendrogram.append(com.flattenDendrogram(tree=tree))
			self._com_ids.append({i:ids[i] for i in xrange(0, len(ids))})


	def getCommunityIds(self, layer):
		"""
		Get the community ids
		"""
		return self._com_ids[layer]


	def getNodes(self, layer, cid):
		"""
		Get the nodes in layer, and communit id
		"""
		return self._ocom[layer].nodesInCommunity(cid)


	def getRootCommunity(self):
		"""
		Get the root nodes of layers community
		"""
		return [c.getRoot() for c in self._ocom]


	def getChildren(self, layer, cid):
		"""
		Get the children of cid in layer
		"""
		return self._dendrogram[layer][cid]['children']


	def getParent(self, layer, cid):
		"""
		Get the parent of cin in layer
		"""
		return self._dendrogram[layer][cid]['parent']


	def getSibling(self, layer, cid):
		"""
		Get the sibling of cid in layer
		"""
		return self._dendrogram[layer][cid]['siblings']


	def checkRoot(self, layer, cid):
		"""
		Check in cid is root in dendrogram
		Returns True if root, otherwis false
		"""
		return self._dendrogram[layer][cid] is not None


	def checkLeaf(self, layer, cid):
		"""
		Check if cid is a leaf in dendrogram
		"""
		return len(self._dendrogram[layer][cid]) == 0


	def getAncestors(self, layer, cid):
		"""
		Get all the ancestors of cid in layer
		"""
		parent = self.getParent(layer, cid)

		if parent is None:
			return []

		return [parent] + self.getAncestors(layer, parent)

	def getDecendents(self, layer, cid):
		"""
		Get all decendents of cid in layer
		"""
		children = self.getChildren(layer, cid)

		if len(children) == 0:
			return []

		return [ children[0], children[1] ] + self.getDecendents(layer, children[0]) + self.getDecendents(layer, children[1])

		

class MABSample(object):
	"""docstring for MABSample"""
	def __init__(self, graph, sample, lweight, lcosts, queried, budget):
		super(MABSample, self).__init__()
		self._sample = sample
		self._graph = graph
		self._lweight = lweight
		self._lcosts = lcosts
		self._queried = queried
		self._budget = budget

		self._bandit = BanditManager(self._graph, self._sample, self._queried)

		self._step = 10
		self._window_size = 10
		self._importance_threshold = 5
		self._scores = []
		self._ppart = None 
		self._bandit.initializeRBandits()


	def _initializeSample(self):
		"""
		Add nodes and edges from 'valid' layers to sample of interest
		"""
		importances = self._lweight.getLayerImportance()

		edges_add = set([])									# edges to add
		edges_sub = set([])									# edges to remove

		for i in xrange(1, len(self._sample)):
			nodes = set(list(self._sample[i].nodes()))
			nodes.difference_update(self._queried[0]) 		# nodes that have not been queried in layer 0

			sg = self._sample[i].subgraph(nodes)

			edges = [frozenset(e) for e in sg.edges()]

			if importances[i] > self._importance_threshold:
				edges_add.update(edges)
			else:
				edges_sub.update(edges)

			# Edges to remove cannot be in edges to add
			edges_sub.difference_update(edges_add)

		# Update sample 0
		self._sample[0].add_edges_from([list(e) for e in edges_add])
		self._sample[0].remove_edges_from([list(e) for e in edges_sub])

		# Remove singletons
		self._sample[0].remove_nodes_from(nx.isolates(self._sample[0]))
		self._ppart = community.best_partition(self._sample[0], randomize=False)
		self._pmod = community.modularity(self._ppart, self._sample[0])


	def _communityUpdateDistance(self, ppart, cpart):
		"""
		Compute the change in community between communities of the
		current sample and previous one
		"""
		return self._evaluation.partitionDistance(ppart, cpart, self._queried[0])


	def _rewards(self, ppart, cpart):
		dist = self._communityUpdateDistance(ppart, cpart)
		imp = self._lweight.getEdgeOverlap(self._bandit.getArms()[0])
		self._past_distances.append(dist)

		return (imp * dist, dist, dist)


	def _checkTerminate(self):
		"""
		Check to see if we should end current iteration of MAB
		"""
		#print('mean', np.mean(self._past_distances))
		if len(self._past_distances) > 5\
		 and np.mean(self._past_distances[-self._window_size:]) < 0.1:
			return True
		return False


	def getScores(self):
		return self._scores


	def sample(self, budget):
		"""
		If budget is none, sample ends after distance is too small
		icost is the initial cost
		mcost is the maximum total cost allowed
		"""
		self._initializeSample()
		self._bandit.initializeCBandits()

		cost = 0

		self._past_distances = []

		while self._budget.getBudgetLeft() > 0:
			# Get next node
			u = self._bandit.getNode()

			if u is False:
				break

			# Update queried and cost
			cost += self._lcosts[0]
			self._budget.consumeBudget(self._lcosts[0])
			self._queried[0].update([u])

			# Previous partition
			sg = self._sample[0].subgraph(self._queried[0])
			ppart = community.best_partition(sg, randomize=False)

			# Query in layer 0
			neighbors = self._graph[0].neighbors(u)
			edges = [(u,v) for v in neighbors]

			# Remove existing edges with u
			if u in self._sample[0].nodes():
				self._sample[0].remove_node(u)

			# Add the new edges
			self._sample[0].add_edges_from(edges)

			# Current partition
			sg = self._sample[0].subgraph(self._queried[0])
			cpart = community.best_partition(sg, randomize=False)

			# Reward for bandit and update
			reward = self._rewards(ppart, cpart)
			self._bandit.updateReward(reward)

			self._lweight.updateEdgeOverlap(self._sample, self._queried[0])

			if self._checkTerminate() and cost > budget:
				break

		

class MultiPlexSampling(object):
	"""docstring for MultiPlexSampling"""
	def __init__(self, fnames=[], budget=0, costs=[]):
		super(MultiPlexSampling, self).__init__()
		self._graph = self._getMultiGraph(fnames)
		self._sample = [nx.Graph() for _ in self._graph]

		self._lcosts = costs
		self._queried = [set([]) for _ in self._graph]

		self._budget = Budget(budget, costs)
		self._lweight = EdgeOverlap(len(self._graph))
		self._roles = RoleManager(self._sample, self._queried)
		self._evaluation = Evaluation(self._graph)

		self._rnd = RNDSample(self._graph, self._sample, self._lcosts,\
		 self._queried, self._budget)
		self._mab = MABSample(self._graph, self._sample, self._lweight,\
		 self._lcosts, self._queried, self._budget)


	def _getMultiGraph(self, fname):
		"""
		Get the differet layers of graph
		"""
		nodes = set([])
		edges = {}

		with open(fname, 'r') as f:
			reader = csv.reader(f, delimiter='\t')
			for row in reader:
				nodes.update([row[0], row[1]])
				if row[2] not in edges:
					edges[row[2]] = []
				edges[row[2]].append((row[0], row[1]))

		mgraph = [None for _ in edges]

		for i in edges:
			mgraph[i] = nx.Graph()
			mgraph[i].add_nodes_from(nodes)
			mgraph[i].add_edges_from(edges[i])

		return mgraph


	def getSample(self):
		budget = self._budget.initializeBudget()

		while self._budget.getBudgetLeft() > 0:
			self._rnd.sample(budget)
			self._mab.sample(budget[0])

			self._budget.updateSlices()
			budget = self._budget.allocateBudget(self._lweight.getEdgeOverlap())

		return self._sample[0]



if __name__ == '__main__':

	fname = 'db/twitter_kp.csv'
	budget = 500
	lcosts = [1, 0.5, 0.5]
	
	mcs = MultiPlexSampling(fname, budget, lcosts)
	sample = mcs.getSample()
