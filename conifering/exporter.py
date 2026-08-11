## exporter.py

import numpy as np
from treebeard.SDT import SDT
import torch
import os

def heap_dfs_mapping(depth):
    """
    mapping torch heap-indexing to conifer DFS-indexing
    returns: dict{heap_id:dfs_id}
    """
    heap_dfs_map = {}
    counter = 0

    def walk(heap_id, current_depth):
        nonlocal counter
        heap_dfs_map[heap_id] = counter
        counter += 1
        if current_depth < depth:
            walk(2 * heap_id + 1, current_depth + 1)
            walk(2 * heap_id + 2, current_depth + 1)
    walk(heap_id=0, current_depth=0)
    return heap_dfs_map

def export_to_hls(tree, n_features, n_classes, n_objects):
    depth    = tree.depth
    n_nodes  = 2 ** (depth + 1) - 1
    n_leaves = 2 ** depth

    ## pull raw trained parameters from tree ##
    beta  = torch.clamp(tree.beta.detach(), max=5.0).cpu().numpy()  # (n_internal, )
    W_raw = tree.inner_nodes[0].weight.detach().cpu().numpy()       # (n_internal, 1+n_features), W_raw[:,0] = bias
    W_full_eff = beta[:, None] * W_raw                              # fold beta into weights

    leaf_logits = tree.leaf_logits.detach().cpu().numpy()           # (n_leaves, n_classes)
    exp = np.exp(leaf_logits - leaf_logits.max(axis=1, keepdims=True))
    leaf_probs_heap = exp / exp.sum(axis=1, keepdims=True)          # (n_leaves, n_classes)

    ## split weights into bias / pt / eta / phi ## 
    bias_heap    = W_full_eff[:, 0]
    weights_heap = W_full_eff[:, 1:]

    w_pt_heap  = weights_heap[:, 0::3]
    w_eta_heap = weights_heap[:, 1::3]
    w_phi_heap = weights_heap[:, 2::3]

    ## heap -> DFS map ##
    mapping = heap_dfs_mapping(depth)

    ## initialise DFS-ordered arrays ##
    bias_dfs  = np.zeros(n_nodes)
    w_pt_dfs  = np.zeros((n_nodes, n_objects))
    w_eta_dfs = np.zeros((n_nodes, n_objects))
    w_phi_dfs = np.zeros((n_nodes, n_objects))

    feature_dfs        = np.full(n_nodes, -1, dtype=int)
    children_left_dfs  = np.full(n_nodes, -1, dtype=int)
    children_right_dfs = np.full(n_nodes, -1, dtype=int)
    parent_dfs         = np.full(n_nodes, -1, dtype=int)

    ## scatter internal node arrays into DFS order ## 
    for heap_id, conifer_id in mapping.items():
        level = int(np.log2(heap_id + 1))
        if level < depth:
            bias_dfs[conifer_id]    = bias_heap[heap_id]
            w_pt_dfs[conifer_id]    = w_pt_heap[heap_id]
            w_eta_dfs[conifer_id]   = w_eta_heap[heap_id]
            w_phi_dfs[conifer_id]   = w_phi_heap[heap_id]

            feature_dfs[conifer_id] = 0  # sentinel: >=0 means internal
            left_heap, right_heap = 2*heap_id+1, 2*heap_id+2
            children_left_dfs[conifer_id]  = mapping[left_heap]
            children_right_dfs[conifer_id] = mapping[right_heap]
            parent_dfs[mapping[left_heap]]  = conifer_id
            parent_dfs[mapping[right_heap]] = conifer_id

    ## leaf_probs into DFS leaf order ##
    leaf_heap_ids = [h for h in mapping if int(np.log2(h+1)) == depth]
    leaf_heap_ids_sorted_by_dfs = sorted(leaf_heap_ids, key=lambda h: mapping[h])
    leaf_local_idx = [h - (2**depth - 1) for h in leaf_heap_ids_sorted_by_dfs]
    leaf_probs_dfs = leaf_probs_heap[leaf_local_idx]

    return dict(
        feature=feature_dfs, bias=bias_dfs,
        weight_pt=w_pt_dfs, weight_eta=w_eta_dfs, weight_phi=w_phi_dfs,
        leaf_probs=leaf_probs_dfs,
        children_left=children_left_dfs, children_right=children_right_dfs,
        parent=parent_dfs,
    )

def format_c_array(arr, fmt="{:.10f}"):
    if arr.ndim == 1:
        return "{" + ",".join(fmt.format(v) for v in arr) + "}"
    return "{" + ",".join(format_c_array(row, fmt) for row in arr) + "}"

def format_c_array_int(arr):
    return "{" + ",".join(str(int(v)) for v in arr) + "}"

def write_parameters_h(params, depth, n_features, n_classes, n_objects, path="firmware/parameters.h"):
    n_nodes  = 2**(depth+1) - 1
    n_leaves = 2**depth

    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    lines = []
    lines.append('#ifndef BDT_PARAMS_H__')
    lines.append('#define BDT_PARAMS_H__')
    lines.append('#include "SDT.h"')
    lines.append('#include "ap_fixed.h"')
    lines.append('')
    lines.append(f'static const int n_features = {n_features};')
    lines.append(f'static const int n_classes = {n_classes};')
    lines.append(f'static const int n_objects = {n_objects};')
    lines.append('')
    lines.append('typedef ap_ufixed<14,12> pt_t;')
    lines.append('typedef ap_fixed<12,5>  eta_t;')
    lines.append('typedef ap_fixed<12,3>  phi_t;')
    lines.append('typedef ap_fixed<13,4>  weight_t;')
    lines.append('typedef ap_fixed<9,5>   logit_t;')
    lines.append('typedef ap_ufixed<6,1>  gate_t;')
    lines.append('typedef ap_ufixed<6,1>  score_t;')
    lines.append('typedef score_t score_arr_t[n_classes];')
    lines.append('')
    lines.append(f'static const BDT::Tree<0, {n_nodes}, {n_leaves}, n_features, n_classes, n_objects, '
                  f'score_t, weight_t, logit_t, gate_t, pt_t, eta_t, phi_t> tree_0_0 = {{')
    lines.append('    ' + format_c_array_int(params['feature']) + ',')
    lines.append('    ' + format_c_array(params['bias']) + ',')
    lines.append('    ' + format_c_array(params['weight_pt']) + ',')
    lines.append('    ' + format_c_array(params['weight_eta']) + ',')
    lines.append('    ' + format_c_array(params['weight_phi']) + ',')
    lines.append('    ' + format_c_array(params['leaf_probs']) + ',')
    lines.append('    ' + format_c_array_int(params['children_left']) + ',')
    lines.append('    ' + format_c_array_int(params['children_right']) + ',')
    lines.append('    ' + format_c_array_int(params['parent']))
    lines.append('};')
    lines.append('#endif')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))