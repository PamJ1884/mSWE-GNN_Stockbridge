"""Batch helpers for multiscale PyG batches, without the lightning dependency.

`adapt_batch_training` and `update_batch_multiscale` are verbatim copies of the
functions of the same name in training/train.py of the original mSWE-GNN
repository (Copyright (c) 2024 Roberto Bentivoglio, MIT License, see LICENSE
at the repository root). They are copied here because training/train.py
imports lightning at module level, which is not available in the Stockbridge
container. training/train.py is left untouched; keep both copies in sync.
"""

import numpy as np
import torch
from torch_geometric.data.batch import Batch


def adapt_batch_training(batch):
    """Corrects batch features for multiscale batches and so that there are also less if/else conditions later on"""
    assert isinstance(batch, Batch), "This function requires a torch_geometric.data.batch.Batch object as input"
    temp = batch.clone()
    temp.node_BC = torch.cat([temp.ptr[i]+temp[i].node_BC for i in range(temp.num_graphs)])
    temp.temporal_res = temp.temporal_res[0]
    temp.type_BC = temp.type_BC[0]
    temp.previous_t = temp.previous_t[0]
    if 'edge_ptr' in temp.keys():
        update_batch_multiscale(temp)
        temp.node_BC_ptr = torch.tensor([torch.where(torch.logical_and(temp.node_ptr[:,0] <= node, node <= temp.node_ptr[:,-1]))[0]
                                                                    for node in temp.node_BC.cpu()])
    else:
        temp.node_BC_ptr = torch.tensor([torch.where(torch.logical_and(temp.ptr[0] <= node, node <= temp.ptr[-1]))[0]
                                                                    for node in temp.node_BC.cpu()])
    return temp


def update_batch_multiscale(batch):
    """Updates the batch for multiscale models, in particular corrects the partitionings needed in MSGNN"""
    batch_edge_ptr = batch.edge_ptr.reshape(batch.num_graphs, -1)
    batch_intra_edge_ptr = batch.intra_edge_ptr.reshape(batch.num_graphs, -1)
    batch_node_ptr = batch.node_ptr.reshape(batch.num_graphs, -1)
    num_scales = batch_intra_edge_ptr.shape[1]

    updated_batch_edge_ptr = [batch_edge_ptr[0]]
    updated_batch_intra_edge_ptr = [batch_intra_edge_ptr[0]]
    updated_batch_node_ptr = [batch_node_ptr[0]]

    for line in batch_edge_ptr[1:]:
        updated_batch_edge_ptr.append(line + updated_batch_edge_ptr[-1].max())

    for line in batch_intra_edge_ptr[1:]:
        updated_batch_intra_edge_ptr.append(line + updated_batch_intra_edge_ptr[-1].max())

    for line in batch_node_ptr[1:]:
        updated_batch_node_ptr.append(line + updated_batch_node_ptr[-1].max())

    updated_batch_edge_ptr = torch.stack(updated_batch_edge_ptr)
    updated_batch_intra_edge_ptr = torch.stack(updated_batch_intra_edge_ptr)
    updated_batch_node_ptr = torch.stack(updated_batch_node_ptr)

    intra_mesh_edge_index = [torch.cat([batch.intra_mesh_edge_index[:,j[0]:j[1]] for j in updated_batch_intra_edge_ptr[:,i:i+2]], 1)
                             for i in range(num_scales-1)]
    edge_index = [torch.cat([batch.edge_index[:,j[0]:j[1]] for j in updated_batch_edge_ptr[:,i:i+2]], 1) for i in range(num_scales)]
    edge_attr = [torch.cat([batch.edge_attr[j[0]:j[1]] for j in updated_batch_edge_ptr[:,i:i+2]]) for i in range(num_scales)]

    batch.node_ptr = updated_batch_node_ptr
    batch.edge_index = torch.cat(edge_index, 1)
    batch.edge_attr = torch.cat(edge_attr)
    batch.edge_ptr = torch.LongTensor(np.cumsum([0] + [edge.shape[1] for edge in edge_index]))
    batch.intra_edge_ptr = torch.LongTensor(np.cumsum([0] + [edge.shape[1] for edge in intra_mesh_edge_index]))
    batch.intra_mesh_edge_index = torch.cat(intra_mesh_edge_index,1)
