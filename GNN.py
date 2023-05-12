import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, GATConv

class CustomGNNLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(CustomGNNLayer, self).__init__()
        self.gcn = GraphConv(in_dim, in_dim)
        self.gatNP = GATConv((in_dim, in_dim), hid_dim, 1)
        self.gatPQ = GATConv((hid_dim, in_dim), out_dim, 1)
        self.gatNP.set_allow_zero_in_degree(True)
        self.gatPQ.set_allow_zero_in_degree(True)

    def forward(self, g):
        with g.local_scope():
            # Get N nodes and create a subgraph with only N nodes and N-to-N edges
            #n_nodes = g.filter_nodes(lambda nodes: nodes.ntype == 'N')
            #n_to_n_subgraph = g.subgraph(n_nodes)
            #n_to_n_subgraph = dgl.to_homogeneous(n_to_n_subgraph)
            n_to_n_subgraph = dgl.edge_type_subgraph(g, [('N', 'N-N', 'N')])
            #n_to_n_subgraph.ndata['h'] = g.nodes['N'].data['feature']

            # Apply a densely connected GCN layer on N-to-N subgraph and update N nodes representation
            #n_to_n_subgraph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_n'))
            g.nodes['N'].data['h'] = self.gcn(n_to_n_subgraph, g.nodes['N'].data['feature'])

            # Update P nodes by max-pooling from N nodes
            g.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h_p'), etype=('N', 'N-P', 'P'))
            g.nodes['P'].data['h'] = F.relu(g.nodes['P'].data['h_p'])

            # Use GAT to get the representation of the whole graph
            # node_features = {
            #     'Q': g.nodes['Q'].data['feature'],
            #     'P': g.nodes['P'].data['h'],
            #     'N': g.nodes['N'].data['h']}
            #g.ndata['h'] = torch.cat([g.nodes['Q'].data['feature'], g.nodes['P'].data['h'], g.nodes['N'].data['h']], dim=0)
            #gat_output = self.gat(g, node_features)
            #g.ndata['h'] = gat_output
            #g.nodes['P'].data['h'] = gatNP()
            g.nodes['P'].data['h'] = self.gatNP(dgl.edge_type_subgraph(g, [('N', 'N-P', 'P')]), (g.nodes['N'].data['h'], g.nodes['P'].data['h']))[..., 0, :]
            gat_out, gat_att = self.gatPQ(dgl.edge_type_subgraph(g, [('P', 'P-Q', 'Q')]), (g.nodes['P'].data['h'], g.nodes['Q'].data['feature']), get_attention=True)
            return gat_out[..., 0, :], gat_att[..., 0, 0].mean(1)

            # Get the final representation of the Q node
            #q_node_id = g.filter_nodes(lambda nodes: nodes.ntype == 'Q')[0].item()
            #final_representation = g.ndata['h'][q_node_id]
            #return final_representation

            # Use GAT layer with multi-headed attention to get the graph representation
            # gat_output, gat_attention_weights = self.gat(g, g.ndata['feature'], get_attention=True)
            # g.ndata['feature'] = gat_output

            # # Get the attention weights between the question node and the page nodes
            # question_node_id = g.filter_nodes(lambda nodes: nodes.ntype == 'Q')[0].item()
            # page_nodes_ids = g.filter_nodes(lambda nodes: nodes.ntype == 'P')
            # question_page_edges = g.edge_ids(question_node_id, page_nodes_ids)

            # # Find the highest attention weight and the corresponding page node
            # attention_weights = gat_attention_weights[question_page_edges].mean(dim=1)
            # highest_attention_index = attention_weights.argmax().item()
            # highest_attention_page_node_id = page_nodes_ids[highest_attention_index]
            # page = highest_attention_page_node_id

            # # Get the final representation of the question node
            # final_representation = g.nodes['Q'].data['feature']
            # #page = g.nodes['page'].data['number']
            # return page, final_representation

    