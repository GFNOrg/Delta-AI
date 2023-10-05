import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import math

class PGM(nn.Module):
    def __init__(self, args, device, adj):
        super(PGM, self).__init__()

        self.device = device
        self.adj = adj
        if args.graph in ["lattice", "torus"]:
            _, chordal_adj = minfill(self.adj.clone().cpu().numpy())
            self.adj = torch.Tensor(chordal_adj).to(device)

        self.n_nodes = args.n_nodes
        self.n_subdags_per_node = args.n_subdags_per_node

        self.K_pa_full = None
        self.K_ch_full = None
        self.top_order_full = None

        self.K_pa = None
        self.K_ch = None
        self.top_order = None

        self.mb_idx, self.max_mb_size = self._get_mb()

    def _get_mb(self):
        max_mb_size = torch.max(torch.sum(self.adj, dim=1)).long().item()
        mb_idx = []

        for i in range(self.adj.shape[0]):
            mb = torch.nonzero(self.adj[i])
            # add node to mb
            mb = torch.cat((mb, torch.Tensor([i]).to(self.device).long().view(-1, 1)), dim=0)
            # Sort
            mb = mb[mb[:, 0].sort()[1]]
            mb_idx.append(mb.view(-1))

        return mb_idx, max_mb_size

    def sample_and_set_fulldag(self):
        max_cliques = max_cardinality_search(self.adj.cpu().numpy())
        sepset_size = get_sepset_size(max_cliques)
        K_pa, K_ch, top_order = markov_to_bayes(max_cliques, sepset_size)
        self.K_pa_full = torch.Tensor(K_pa).long().to(self.device)
        self.K_ch_full = torch.Tensor(K_ch).long().to(self.device)
        self.top_order_full = torch.Tensor(top_order).long().to(self.device)

    def sample_fulldag(self):
        max_cliques = max_cardinality_search(self.adj.cpu().numpy())
        sepset_size = get_sepset_size(max_cliques)
        K_pa, K_ch, top_order = markov_to_bayes(max_cliques, sepset_size)
        K_pa_full = torch.Tensor(K_pa).long().to(self.device)
        K_ch_full = torch.Tensor(K_ch).long().to(self.device)
        top_order_full = torch.Tensor(top_order).long().to(self.device)
        return K_pa_full, K_ch_full, top_order_full

    def sample_and_set_subdags(self, ilist):
        top_order_list = []
        K_pa_list = []
        K_ch_list = []
        max_K_pa_size = 0
        max_K_ch_size = 0

        for i in ilist:
            mb = self.mb_idx[i]
            # Get sub-graph of {V_i} + Mb_i
            sub_adj_i = self.adj[mb,:][:,mb]
            # Maximal cardinality search
            max_cliques = max_cardinality_search(sub_adj_i.cpu().numpy())
            # Get sepset sizes
            sepset_size = get_sepset_size(max_cliques)

            sub_K_pa_i, sub_K_ch_i, sub_top_order_i = \
                    markov_to_bayes(max_cliques, sepset_size, batch_size=self.n_subdags_per_node)
            sub_K_pa_i = torch.Tensor(sub_K_pa_i).to(self.device).long()
            sub_K_ch_i = torch.Tensor(sub_K_ch_i).to(self.device).long()
            sub_top_order_i = torch.Tensor(sub_top_order_i).to(self.device).long()

            # switch to global index
            sub_top_order_i = mb[sub_top_order_i]
            # Switch to global index sub_K_pa_i[:,1:] and keep the first column as is
            sub_K_pa_i = mb[sub_K_pa_i]
            sub_K_ch_i = mb[sub_K_ch_i]

            top_order_list.append(sub_top_order_i)
            K_pa_list.append(sub_K_pa_i)
            K_ch_list.append(sub_K_ch_i)

            if sub_K_pa_i.shape[1] > max_K_pa_size:
                max_K_pa_size = sub_K_pa_i.shape[1]
            if sub_K_ch_i.shape[1] > max_K_ch_size:
                max_K_ch_size = sub_K_ch_i.shape[1]

        n_nod = self.n_nodes
        n_spn = self.n_subdags_per_node

        top_order = torch.full((n_nod * n_spn, self.max_mb_size + 1), -1).long().to(self.device)
        K_pa = torch.full((n_nod * n_spn, max_K_pa_size, 2), -2).long().to(self.device)
        K_ch = torch.full((n_nod * n_spn, max_K_ch_size, 2), -2).long().to(self.device)

        for i in range(n_nod):
            K_pa[i*n_spn:(i+1)*n_spn, :K_pa_list[i].shape[1]] = K_pa_list[i]
            K_ch[i*n_spn:(i+1)*n_spn, :K_ch_list[i].shape[1]] = K_ch_list[i]
            top_order[i*n_spn:(i+1)*n_spn, :top_order_list[i].shape[1]] = top_order_list[i]

        self.K_pa = K_pa
        self.K_ch = K_ch
        self.top_order = top_order

    # Returns the indexes of parents of itself for each instance in the batch
    def get_idx_from_K(self, K, itself):
        if K.shape[0] == 1:
            batch_idx = torch.nonzero(itself.unsqueeze(1) == K[0, :, 0])
            instance_idx = batch_idx[:, 0]
            K_idx = K[0, batch_idx[:, 1], 1]
        else:
            batch_idx = torch.nonzero(itself.unsqueeze(1) == K[..., 0])
            instance_idx = batch_idx[:, 0]
            K_idx = K[instance_idx, batch_idx[:, 1], 1]
        return instance_idx, K_idx

def markov_to_bayes(max_cliques, sepset_size, batch_size=1):
    """
    Given the adjacency matrix of a Markov network, finds a Bayes net that is an I-Map for the Markov network.
    If there are many possible I-Maps, one of them is returned with uniform probability.
    """

    # Randomly choose root index
    K_pa_all = []
    K_ch_all = []
    var_order_all = []
    for i in range(batch_size):
        # Permute the max_cliques order and the corresponding rows/columns in sepset_size to get different max span trees
        perm = np.random.permutation(len(max_cliques))
        sepset_size = sepset_size[perm, :]
        sepset_size = sepset_size[:, perm]
        max_cliques = [max_cliques[i] for i in perm]
        # Construct the clique tree
        max_span_tree_adj = max_span_tree(sepset_size)
        root = np.random.choice(len(max_cliques))
        ctree_children = max_span_tree_to_ctree(max_span_tree_adj, root)
        # Permute the ordering of the nodes in each clique to get different DAGs
        max_cliques = [np.random.permutation(list(clique)) for clique in max_cliques]
        # Use clique tree to create P-Map for chordal graph
        K_pa, K_ch, var_order = ctree_to_dag(max_cliques, ctree_children)
        K_pa_all.append(K_pa)
        K_ch_all.append(K_ch)
        var_order_all.append(var_order)

    return K_pa_all, K_ch_all, var_order_all

def minfill(adj):
    """
    return the elimination order (a list of indices) as well
    as the resulting chordal graph based on min_fill heuristic
    chordal_adj[i,j] = 1 iff (i,j) are connected.
    The diagonal of chordal_adj is zero.
    """

    idx_list = []
    vdim = adj.shape[0]
    d = int(math.sqrt(vdim))
    i, j = 0, 0
    up = False
    down = False
    first = True
    for _ in range(vdim):
        idx = i*d + j
        idx_list.append(idx)

        if i == d-1 and j == 0:
            first = False

        if first:
            if up:
                i -= 1
                j += 1
                if i == 0:
                    up = False
                continue
            if down:
                i += 1
                j -= 1
                if j == 0:
                    down = False
                continue
            if i == 0:
                j += 1
                down = True
                continue
            if j == 0:
                i += 1
                up = True
            else:
                raise NotImplementedError
        else:
            if up:
                i -= 1
                j += 1
                if j == d-1:
                    up = False
                continue
            if down:
                i += 1
                j -= 1
                if i == d-1:
                    down = False
                continue
            if i == d-1:
                j += 1
                up = True
                continue
            if j == d-1:
                i += 1
                down = True
            else:
                raise NotImplementedError
    i_remaining = np.array(idx_list)

    order = []

    while len(i_remaining) > 0:
        n_fill = []
        for i in i_remaining:
            i_inter = np.nonzero(adj[i, :])[0]
            i_removed = np.isin(i_inter, order)  # No interaction with previously removed node
            i_inter = i_inter[~i_removed]
            if len(i_inter) < 2:
                # No interactions => no fill edges
                n_fill.append(0)
                continue
            sub_adj = adj[i_inter, :][:, i_inter]  # Sub-matrix between the interaction nodes
            i_sub_adj_tril = np.tril_indices_from(sub_adj, -1)
            zero_mask = (sub_adj[i_sub_adj_tril] == 0)
            n_fill.append(zero_mask.sum())  # nodes without interactions will need fill edges

        i_min_remaining = np.argmin(n_fill)
        i_min_node = i_remaining[i_min_remaining]
        order.append(i_min_node)
        i_remaining = np.delete(i_remaining, i_min_remaining)
        i_inter = np.nonzero(adj[i_min_node, :])[0]
        i_removed = np.isin(i_inter, order)  # No interaction with previously removed node
        i_inter = i_inter[~i_removed]
        if len(i_inter) >= 2:  # No edges to fill otherwise
            i_sub_adj_off_diag = np.array([[i, j] for i in i_inter for j in i_inter if i != j])
            i_sub_adj_off_diag = (i_sub_adj_off_diag[:, 0], i_sub_adj_off_diag[:, 1])
            adj[i_sub_adj_off_diag] = 1

    return order, adj

def max_cardinality_search(mask):
    """
    mask is the adjacency matrix for a chordal graph
    this method returns a list of sets: the set of nodes in maximal cliques
    """
    n = mask.shape[0]
    cliques = [[]]  # maintains the list of cliques
    last_mark = -1  # number of marked neighbors for prev. node
    marks = [[] for i in range(n)]  # a set tracking the marked neighbors of each node
    mark_size = np.zeros(n)  # number of marked neighbors for each node
    remaining = list(range(n))
    for _ in reversed(range(n)):
        node = remaining[np.argmax(mark_size[remaining])]
        if mark_size[node] <= last_mark:  # moving into a new clique
            cliques.append(marks[node] + [node])
        else:  # add it to the last clique
            cliques[-1].append(node)
        nb_node = np.nonzero(mask[node, :])[0]  # neighbors of node
        for nb in nb_node:  # update the marks for neighbors
            marks[nb].append(node)
            mark_size[nb] += 1
        last_mark = mark_size[node]
        remaining.remove(node)
    sorted_cliques = [set(c) for c in cliques]

    return sorted_cliques

def get_sepset_size(cliques):
    """
    Given a list of maximal cliques in a clique tree, returns a matrix of the size of the sepsets.
    """
    num_cliques = len(cliques)
    sepset_size = np.zeros((num_cliques, num_cliques))

    for i in range(num_cliques):
        cl1 = cliques[i]
        for j in range(i + 1, num_cliques):
            cl2 = cliques[j]
            sepset_size[i, j] = sepset_size[j, i] = max(len(cl1 & cl2), 0.1)

    return sepset_size

def max_span_tree_to_ctree(adj, root=0):
    """
    Given a max spanning tree adjacency, and root node, returns a clique tree.
    """
    to_visit = set([root])
    n = adj.shape[0]
    rest = set(range(n)) - to_visit
    children = {}
    while len(to_visit) > 0:
        current = to_visit.pop()
        nexts = set(np.nonzero(adj[current, :])[0]).intersection(rest)
        children[current] = frozenset(nexts)
        to_visit.update(nexts)
        rest.difference_update(nexts)

    return children

def max_span_tree(sepset_size):
    """
    Given a matrix of sepset sizes and root node, returns a maximum spanning tree.
    """
    max_span_tree = minimum_spanning_tree(csr_matrix(-sepset_size)).toarray().astype(int)
    max_span_tree_adj = (np.maximum(-np.transpose(max_span_tree.copy()), -max_span_tree) > 0).astype(int)

    return max_span_tree_adj

def ctree_to_dag(cliques, ctree_children):
    """
    Given a clique tree, directs the edges of the original Markov network to create a Bayes net I-Map
    """
    var_clique_dict = {}
    var_order = []

    # Create the topological ordering such that:
    # 1. fully connected within cliques (uses the ordering within cliques)
    # 2. from root to leaves between cliques (uses the ordering within ctree_children)
    i_root_clique = next(iter(ctree_children.keys()))
    root_clique = cliques[i_root_clique]
    for node in root_clique:
        var_clique_dict[node] = root_clique
        var_order.append(node)
    for i_clique, i_children in ctree_children.items():
        for i_child_clique in i_children:
            child_clique = cliques[i_child_clique]
            for node in child_clique:
                if var_clique_dict.get(node) is None:
                    var_clique_dict[node] = child_clique
                    var_order.append(node)

    # create a Bayes net by adding edges respecting the topological ordering between and within cliques
    K_pa = []
    K_ch = []
    for order_index in range(len(var_order)):
        node = var_order[order_index]
        node_parents = (set(var_clique_dict[node]) - set([node])).intersection(set(var_order[:order_index]))
        for parent in node_parents:
            K_pa.append([node, parent])
            K_ch.append([parent, node])

    return K_pa, K_ch, var_order
