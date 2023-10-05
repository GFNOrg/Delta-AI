import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

class PGM(nn.Module):
    def __init__(self, args, adj):
        super(PGM, self).__init__()

        self.device = args.device

        self.adj = adj

        self.vdim = args.vdim
        self.xdim = args.xdim

        self.cond_indep_H_F = True if args.q_objective == "mean_field" else False

        self.batchsz = args.batchsz
        self.n_nodes = args.n_nodes
        self.repeat_nodes = args.repeat_nodes
        self.n_subdags_per_node = args.n_subdags_per_node
        self.n_fulldags_per_batch = args.n_fulldags_per_batch

        # Chordalize
        chordal_adj = minfill(self.adj.clone().cpu().numpy())
        self.adj = torch.Tensor(chordal_adj).to(self.device)

        self.K_pa = {"full": {"p": None, "q": None}, "partial": {"p": None, "q": None}}
        self.K_ch = {"full": {"p": None, "q": None}, "partial": {"p": None, "q": None}}
        self.top_order = {"full": {"p": None, "q": None}, "partial": {"p": None, "q": None}}

        self.adj_H = self.adj[self.xdim:, self.xdim:]
        self.adj_X = self.adj[:self.xdim, :self.xdim]

        # Store edges between X and H
        self.K_pa_XH = self.adj[:self.xdim, self.xdim:].nonzero().unsqueeze(0)
        self.K_pa_XH[:, :, 1] = self.K_pa_XH[:, :, 1] + self.xdim

        self.sample_and_set_fulldag(batchsz=1)

        if args.q_objective == "gibbs":
            # get the set of arguments for conditionals in B involving H
            self.H_cond = []
            for i in range(self.xdim, self.vdim):
                # itself = torch.tensor([i]).to(self.device)
                cond_list_i = self.K_ch["full"]["p"][0, (self.K_ch["full"]["p"] == i)[0, :, 0], 1].tolist()
                cond_list_i.append(i)
                self.H_cond.append(cond_list_i)
        else:
            self.H_cond = None

        self.mb_idx, self.mb_idx_H, self.mb_idx_X, self.max_mb_size = self._get_mb()

        print("Successfully initialized PGM.")

    def _get_mb(self):
        max_mb_size = torch.max(torch.sum(self.adj, dim=1)).long().item()
        mb_idx = []
        mb_idx_H = []
        mb_idx_X = []

        for i in range(self.adj.shape[0]):
            mb = torch.nonzero(self.adj[i])
            mb = torch.cat((mb, torch.Tensor([i]).to(dtype=torch.int64, device=self.device).view(-1, 1)), dim=0)
            # Sort
            mb = mb[mb[:, 0].sort()[1]]
            mb = mb.view(-1)
            mb_idx.append(mb)

            mb_idx_H.append(mb[mb >= self.xdim])
            mb_X = mb[mb < self.xdim]
            for j in mb_X:
                mb_idx_X.append([i, j])

        mb_idx_X = torch.Tensor(mb_idx_X).to(dtype=torch.int64, device=self.device).unsqueeze(0)

        return mb_idx, mb_idx_H, mb_idx_X, max_mb_size

    def sample_and_set_fulldag(self, batchsz=1):

        # DAG for q
        if self.cond_indep_H_F:
            K_pa_F = torch.zeros((batchsz, 0, 2), dtype=torch.int64, device=self.device)
            top_order_F = torch.arange(self.vdim-self.xdim).unsqueeze(0).to(dtype=torch.int64, device=self.device)
            if batchsz > 1:
                top_order_F = top_order_F.expand(batchsz, -1)
        else:
            K_pa_F, top_order_F = sample_dag_pmap(self.adj_H, batch_size=batchsz)
            K_pa_F += self.xdim

        top_order_F += self.xdim

        # DAG for p
        top_order_X = torch.arange(self.xdim).unsqueeze(0).to(dtype=torch.int64, device=top_order_F.device)
        if batchsz > 1:
            top_order_X = top_order_X.expand(batchsz, -1)

        if self.cond_indep_H_F:
            K_pa_H, top_order_H = sample_dag_pmap(self.adj_H, batch_size=batchsz)
            K_pa_H += self.xdim
            top_order_H += self.xdim
            top_order_B = torch.cat((top_order_H, top_order_X), dim=1)
        else:
            top_order_B = torch.cat((top_order_F, top_order_X), dim=1)

        if batchsz > 1:
            K_pa_XH = self.K_pa_XH.expand(batchsz, -1, -1)
        else:
            K_pa_XH = self.K_pa_XH
        if self.cond_indep_H_F:
            K_pa_B = torch.cat((K_pa_H, K_pa_XH), dim=1)
        else:
            K_pa_B = torch.cat((K_pa_F, K_pa_XH), dim=1)

        if self.n_fulldags_per_batch > 1:
            K_pa_B = torch.repeat_interleave(K_pa_B, self.batchsz // self.n_fulldags_per_batch, dim=0)
            top_order_B = torch.repeat_interleave(top_order_B, self.batchsz // self.n_fulldags_per_batch, dim=0)
            K_pa_F = torch.repeat_interleave(K_pa_F, self.batchsz // self.n_fulldags_per_batch, dim=0)
            top_order_F = torch.repeat_interleave(top_order_F, self.batchsz // self.n_fulldags_per_batch, dim=0)

        self.K_pa["full"]["p"] = K_pa_B
        self.K_ch["full"]["p"] = K_pa_B[:, :, [1, 0]]
        self.top_order["full"]["p"] = top_order_B

        self.K_pa["full"]["q"] = K_pa_F
        self.K_ch["full"]["q"] = K_pa_F[:, :, [1, 0]]
        self.top_order["full"]["q"] = top_order_F

    def sample_and_set_subdags(self, ilist):
        K_pa , top_order = self.sample_subdags(ilist)

        for d in ["q", "p"]:
            self.K_pa["partial"][d] = K_pa[d]
            self.K_ch["partial"][d] = K_pa[d][:, :, [1, 0]]
            self.top_order["partial"][d] = top_order[d]

    def sample_subdags(self, ilist):

        top_order_list = {"q": [], "p": []}
        K_pa_list = {"q": [], "p": []}
        max_K_pa_size = {"q": 0, "p": 0}
        top_order = {"q": None, "p": None}
        K_pa = {"q": None, "p": None}

        for i in ilist:
            mb = self.mb_idx_H[i]
            # Sample sub-DAG I-map for the sub-graph involving the node i and its Markov blanket
            sub_K_pa_i, sub_top_order_i = sample_dag_pmap(self.adj[mb, :][:, mb],
                                                          batch_size=self.n_subdags_per_node)

            # switch to global index
            sub_top_order_i = mb[sub_top_order_i]
            sub_K_pa_i = mb[sub_K_pa_i]

            top_order_list["q"].append(sub_top_order_i)
            K_pa_list["q"].append(sub_K_pa_i)

            if sub_K_pa_i.shape[1] > max_K_pa_size["q"]:
                max_K_pa_size["q"] = sub_K_pa_i.shape[1]

            # Find all edges between X and H
            mb_XH = self.mb_idx[i]
            x_mask = mb_XH < self.xdim
            mb_X = mb_XH[x_mask]
            # direct them towards X
            sub_K_pa_XH_i = self.adj[mb_X, :][:, mb_XH[~x_mask]].nonzero().unsqueeze(0)
            sub_top_order_X_i = mb_X.unsqueeze(0)
            # Switch to global index
            if sub_K_pa_XH_i.shape[1] > 0:
                sub_K_pa_XH_i[:, :, 0] = mb_XH[x_mask][sub_K_pa_XH_i[:, :, 0]]
                sub_K_pa_XH_i[:, :, 1] = mb_XH[~x_mask][sub_K_pa_XH_i[:, :, 1]]

                if self.n_subdags_per_node > 1:
                    sub_K_pa_XH_i = sub_K_pa_XH_i.expand(self.n_subdags_per_node, -1, -1)
                    sub_top_order_X_i = sub_top_order_X_i.expand(self.n_subdags_per_node, -1)

                sub_K_pa_i = torch.cat((sub_K_pa_i, sub_K_pa_XH_i), dim=1)
                sub_top_order_i = torch.cat((sub_top_order_i, sub_top_order_X_i), dim=1)

            top_order_list["p"].append(sub_top_order_i)
            K_pa_list["p"].append(sub_K_pa_i)
            if sub_K_pa_i.shape[1] > max_K_pa_size["p"]:
                max_K_pa_size["p"] = sub_K_pa_i.shape[1]

        for key in top_order_list.keys():
            top_order[key] = torch.full((self.n_nodes * self.n_subdags_per_node, self.max_mb_size + 1), -1).to(dtype=torch.int64, device=self.device)
            K_pa[key] = torch.full((self.n_nodes * self.n_subdags_per_node, max_K_pa_size[key], 2), -2).to(dtype=torch.int64, device=self.device)

            for i in range(self.n_nodes):
                n_edges = K_pa_list[key][i].shape[1]
                if n_edges > 0:
                    K_pa[key][i * self.n_subdags_per_node:(i + 1) * self.n_subdags_per_node, :n_edges] = K_pa_list[key][i]
                top_order[key][i * self.n_subdags_per_node:(i + 1) * self.n_subdags_per_node, :top_order_list[key][i].shape[1]] = top_order_list[key][i]

            if self.repeat_nodes > 1:
                K_pa[key] = torch.repeat_interleave(K_pa[key], self.repeat_nodes, dim=0)
                top_order[key] = torch.repeat_interleave(top_order[key], self.repeat_nodes, dim=0)

        return K_pa, top_order

def sample_dag_pmap(adj, batch_size=1):
    """
    Given the adjacency matrix of a chordal Markov network, samples a DAG that is an P-map for the Markov network.
    """

    max_cliques = max_cardinality_search(adj.cpu().numpy())
    sepset_size = get_sepset_size(max_cliques)
    device = adj.device

    K_pa_all = []
    top_order_all = []
    for i in range(batch_size):
        # Construct the clique tree
        max_span_tree_adj = max_span_tree(sepset_size)
        root = np.random.choice(len(max_cliques))
        ctree_children = max_span_tree_to_ctree(max_span_tree_adj, root)
        # Permute the ordering of the nodes in each clique to get different I-maps
        max_cliques = [np.random.permutation(list(clique)) for clique in max_cliques]
        # Use clique tree to direct the edges
        K_pa, var_order = ctree_to_dag(max_cliques, ctree_children)
        K_pa_all.append(K_pa)
        top_order_all.append(var_order)

    K_pa_all = torch.tensor(K_pa_all).to(dtype=torch.int64, device=device)
    top_order_all = torch.tensor(top_order_all).to(dtype=torch.int64, device=device)

    return K_pa_all, top_order_all

def minfill(adj):
    """
    min-fill algorithm to find a chordal completion of a graph with adjacency adj
    """

    i_remaining = np.arange(adj.shape[0])
    order = []
    while len(i_remaining) > 0:
        n_fill = []
        for i in i_remaining:
            # Check if relation to previously eliminated node
            i_rel = np.nonzero(adj[i, :])[0]
            i_removed = np.isin(i_rel, order)
            i_rel = i_rel[~i_removed]
            if len(i_rel) < 2:
                n_fill.append(0)
                continue

            # Add fill edges between related nodes
            sub_adj = adj[i_rel, :][:, i_rel]
            i_sub_adj_tril = np.tril_indices_from(sub_adj, -1)
            zero_mask = (sub_adj[i_sub_adj_tril] == 0)
            n_fill.append(zero_mask.sum())

        # Eliminate node with minimum number of fill edges
        i_min_remaining = np.argmin(n_fill)
        i_min_node = i_remaining[i_min_remaining]
        order.append(i_min_node)
        i_remaining = np.delete(i_remaining, i_min_remaining)
        i_rel = np.nonzero(adj[i_min_node, :])[0]
        i_removed = np.isin(i_rel, order)
        i_rel = i_rel[~i_removed]
        if len(i_rel) >= 2:
            i_sub_adj_off_diag = np.array([[i, j] for i in i_rel for j in i_rel if i != j])
            i_sub_adj_off_diag = (i_sub_adj_off_diag[:, 0], i_sub_adj_off_diag[:, 1])
            adj[i_sub_adj_off_diag] = 1

    return adj

def max_cardinality_search(adj):
    """
    From the adjacency of a chordal graph, returns a list of maximal cliques
    """
    n = adj.shape[0]
    cliques = [[]]
    last_mark = -1
    marks = [[] for _ in range(n)]
    mark_size = np.zeros(n)
    remaining = list(range(n))
    for _ in reversed(range(n)):
        node = remaining[np.argmax(mark_size[remaining])]
        if mark_size[node] <= last_mark:
            cliques.append(marks[node] + [node])
        else:
            cliques[-1].append(node)
        nb_node = np.nonzero(adj[node, :])[0]
        for nb in nb_node:
            marks[nb].append(node)
            mark_size[nb] += 1
        last_mark = mark_size[node]
        remaining.remove(node)
    sorted_cliques = [set(c) for c in cliques]

    return sorted_cliques

def get_sepset_size(cliques):
    """
    Given a list of maximal cliques in a clique tree, returns a matrix with the size of the sepsets.
    """
    num_cliques = len(cliques)
    sepset_size = np.zeros((num_cliques, num_cliques))

    for i in range(num_cliques):
        clique_i = cliques[i]
        for j in range(i + 1, num_cliques):
            clique_j = cliques[j]
            sepset_size[i, j] = sepset_size[j, i] = max(len(clique_i & clique_j), 0.1)

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
    Given a clique tree, directs the edges of the original Markov network to sample an I-Map
    """
    var_clique_dict = {}
    top_order = []

    # Create the topological ordering such that:
    # 1. fully connected within cliques (uses the ordering within cliques)
    # 2. from root to leaves between cliques (uses the ordering within ctree_children)
    i_root_clique = next(iter(ctree_children.keys()))
    root_clique = cliques[i_root_clique]
    for node in root_clique:
        var_clique_dict[node] = root_clique
        top_order.append(node)
    for i_clique, i_children in ctree_children.items():
        for i_child_clique in i_children:
            child_clique = cliques[i_child_clique]
            for node in child_clique:
                if var_clique_dict.get(node) is None:
                    var_clique_dict[node] = child_clique
                    top_order.append(node)

    # Orient the edges according to the topological ordering
    K_pa = []
    for order_index in range(len(top_order)):
        node = top_order[order_index]
        node_parents = (set(var_clique_dict[node]) - set([node])).intersection(set(top_order[:order_index]))
        for parent in node_parents:
            K_pa.append([node, parent])

    return K_pa, top_order

def get_idx_from_K(K, itself):
    """
    Returns the indexes of parents or children of itself for each instance in the batch
    """
    if K.shape[0] == 1:
        batch_idx = torch.nonzero(itself.unsqueeze(1) == K[0, :, 0])
        instance_idx = batch_idx[:, 0]
        K_idx = K[0, batch_idx[:, 1], 1]
    else:
        batch_idx = torch.nonzero(itself.unsqueeze(1) == K[..., 0])
        instance_idx = batch_idx[:, 0]
        K_idx = K[instance_idx, batch_idx[:, 1], 1]
    return instance_idx, K_idx