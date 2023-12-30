import pandas as pd
import numpy as np
import time
import torch
import math
import pickle
from torchppr.api import personalized_page_rank
import networkx as nx
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def load_obj( name ):
    with open( name , 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def subnet_of_rwr(nodes,adj_normalized,seeds,threshold,alpha=0.3):
    rwr=personalized_page_rank(adj=adj_normalized,indices=seeds,alpha=alpha)
    rwr = rwr.numpy().squeeze()
    arr = np.argsort(-rwr)
    arr1 = -np.sort(-rwr)
    ids = arr[arr1>threshold]
    id_score = arr1[arr1>threshold]
    top_mgs = nodes.iloc[ids]
    rwr_sub_net = pd.DataFrame({'name':top_mgs['name'],'score':id_score})
    return  rwr_sub_net

def load_omics_data(cancer):
    ppnet, ppnodes = load_net_file('./data/network/PPIcheng_proced.txt')
    rnaseq = pd.read_table(filepath_or_buffer='./data/tcga/%s/RNAseq.txt' % cancer, sep='\t', header=0, index_col=0)
    rnaseq.columns = [s.replace('.','-') for s in rnaseq.columns.values.tolist()]
    mut = pd.read_table(filepath_or_buffer='./data/tcga/%s/Mutation.txt' % cancer, sep='\t', header=0, index_col=0)
    mut.columns = [s.replace('.','-') for s in mut.columns.values.tolist()]

    rna_nodes = pd.DataFrame({'gene':rnaseq.index.values.tolist()})
    rna_nodes.drop_duplicates(subset=['gene'],keep='first', inplace=True)
    rna_nodes = pd.merge(left=ppnodes,right=rna_nodes,how='left',left_on='name',right_on='gene')
    rna_nodes.dropna(how='any',inplace=True)
    ppnet_sub = extract_edges_on_nodes(ppnet,rna_nodes.iloc[:,[0]])
    ppnodes_sub = pd.concat([ppnet_sub['source'], ppnet_sub['target']])
    ppnodes_sub.drop_duplicates(keep='first', inplace=True)
    ppnodes_sub = pd.DataFrame(ppnodes_sub, columns=['name'])

    rnaseq = rnaseq.loc[ppnodes_sub['name'],:]
    mut = build_mut_mat_by_nodes(mut,ppnodes_sub)
    return mut, rnaseq, ppnet_sub, ppnodes_sub

def calc_personalized_network_weight(ppnet,rnaseq,samp_id):
    samp_rna = rnaseq[samp_id]
    sour_rna = ppnet['source'].values.tolist()
    targ_rna = ppnet['target'].values.tolist()
    samp_sour_rna = samp_rna[sour_rna]
    samp_targ_rna = samp_rna[targ_rna]
    weight_ls = np.multiply(np.array(samp_sour_rna),np.array(samp_targ_rna))
    return weight_ls


def cal_outlying_gene_matrix(df,std_threshold):
    """
    For each gene, we assume the expressions across all the patients are normally distributed.
    The outliers for gene i are defined as those whose values are outside the two-standard deviation
    range of the expression values of gene i across all the patients. --by DriverNet
    """
    expr_df_T = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    test=StandardScaler().fit_transform(expr_df_T)
    expr_df = pd.DataFrame(test.T, index=df.index, columns=df.columns)
    expr_df_abs=expr_df.abs()
    outlier_df = (expr_df_abs>std_threshold).astype(np.int32)
    return outlier_df

def build_mut_mat_by_nodes(mut,nodes):
    mut_dic = dict()
    for g in nodes['name'].tolist():
        if g in mut.index.values.tolist():
            mut_dic[g] = mut.loc[g].tolist()
        else:
            mut_dic[g] = [0 for i in range(0,mut.shape[1])]
    return pd.DataFrame(mut_dic,index=mut.columns).transpose()

def extract_edges_on_nodes(net,subnodes):
    net = pd.merge(left=net, right=subnodes, how='left', left_on='source', right_on='name')
    net.dropna(how='any', inplace=True)
    net = pd.merge(left=net, right=subnodes, how='left', left_on='target', right_on='name')
    net.dropna(how='any', inplace=True)
    net = net[['source', 'target']]
    return net

def load_net_file(file):
    ppnet = pd.read_table(filepath_or_buffer=file, sep='\t', header=None,
                          index_col=None, names=['source', 'target'])
    ppnodes = pd.concat([ppnet['source'], ppnet['target']])
    ppnodes.drop_duplicates(keep='first', inplace=True)
    ppnodes = pd.DataFrame(ppnodes, columns=['name'])
    return ppnet,ppnodes

def convert_edge_to_adj(edge_df,nodes,weighted=False):
    G = nx.Graph()
    G.add_nodes_from(nodes['name'].values.tolist())
    if weighted:
        G.add_edges_from([(edge_df.iloc[i, 0],
                           edge_df.iloc[i, 1],
                           {'weight': edge_df.iloc[i, 2]})
                          for i in range(0, edge_df.shape[0])])
        adj = nx.adjacency_matrix(G, nodelist=np.array(nodes['name']), weight='weight').todense()
    else:
        G.add_edges_from([(edge_df.iloc[i, 0],
                           edge_df.iloc[i, 1])
                          for i in range(0, edge_df.shape[0])])
        adj = nx.adjacency_matrix(G, nodelist=np.array(nodes['name'])).todense()
    return adj

def create_incidence_matrix(samp_mut_target_dic,mut_in_targets=False):
    samp_mut = list(samp_mut_target_dic.keys())
    nodes = []
    for k in samp_mut_target_dic.keys():
        for n in samp_mut_target_dic[k]:
            if n not in nodes:
                nodes.append(n)
    inc_mtx=np.zeros((len(nodes),len(samp_mut_target_dic)))
    inc_mtx= pd.DataFrame(inc_mtx,index=nodes,columns=samp_mut_target_dic.keys(),dtype=int)
    for k in samp_mut_target_dic.keys():
        targets = samp_mut_target_dic[k]
        inc_mtx[k].loc[targets]=1
    if not mut_in_targets:
        drop_ls = []
        for m in samp_mut:
            if m in inc_mtx.index:
                drop_ls.append(m)
        inc_mtx.drop(drop_ls,axis=0,inplace=True)
    else:
        for m in samp_mut:
            if m in inc_mtx.index:
                inc_mtx.loc[m,m]=1
            else:
                inc_mtx.loc[m] = [0 for _ in range(0,len(samp_mut))]
                inc_mtx.loc[m,m]=1
    return inc_mtx

def calc_weighted_jaccard_centrality(inc_mtx):
    samp_mut = inc_mtx.columns.values.tolist()
    jac_mtx = np.zeros((len(samp_mut),len(samp_mut)))
    jac_mtx = pd.DataFrame(jac_mtx,index=samp_mut,columns=samp_mut)
    for i in range(0, len(samp_mut)):
        for j in range(i + 1, len(samp_mut)):
            u1 = inc_mtx[samp_mut[i]][inc_mtx[samp_mut[i]]==1].index.values.tolist()
            u2 = inc_mtx[samp_mut[j]][inc_mtx[samp_mut[j]]==1].index.values.tolist()
            if len(u1) > 0 and len(u2) > 0:
                jac_mtx.loc[samp_mut[i],samp_mut[j]] = len(list(set(u1) & set(u2))) / (len(list(set(u1 + u2))))
            else:
                jac_mtx.loc[samp_mut[i], samp_mut[j]] = 0
    jac_mtx = jac_mtx + jac_mtx.transpose()
    jac_cen = jac_mtx.sum(axis=1)/(len(samp_mut)-1)
    weigh = pd.Series([inc_mtx[m].sum() for m in samp_mut],index=samp_mut)
    jac_cen_weig = jac_cen.mul(weigh)
    return jac_cen_weig

def HeRW_on_individual_hypergraph(inc_mtx):
    # Hypergraph random walk
    e = np.array(inc_mtx)
    C_hat = np.eye(e.shape[1])
    jcw = calc_weighted_jaccard_centrality(inc_mtx)
    for i in range(0,C_hat.shape[0]):
        C_hat[i,i] = jcw.iloc[i]
    K_H = np.matmul(np.matmul(e, C_hat), e.T)
    K_H[np.eye(K_H.shape[0],dtype=np.bool_)] = 0
    d_H = np.sum(K_H,axis=1)
    p = d_H/np.sum(K_H)
    weight_e = p.reshape(-1,1) * e
    weight_e = pd.DataFrame(weight_e,index=inc_mtx.index,columns=inc_mtx.columns)
    mut_score = weight_e.sum(axis=0)
    mut_score = mut_score.sort_values(ascending=False)
    return p,mut_score

def dif_path_frequence(gene_path,mg1,mg2):
    if (mg1 in gene_path.index.values.tolist()) and (mg2 in gene_path.index.values.tolist()):
        path_mg1 = gene_path.loc[mg1] == 1
        path_mg1 = path_mg1[path_mg1==1].index.values.tolist()
        path_mg2 = gene_path.loc[mg2] == 1
        path_mg2 = path_mg2[path_mg2 == 1].index.values.tolist()
        return len(path_mg1) + len(path_mg2)
    else:
        return 0


def find_heavy_subgraph(node_scores,edge_weights,penalty,num_nodes):
    '''
    This is a variation of the NP-Hard problem called the Maximum Weight Subgraph problem.
    Here, you seek to find a subgraph (in your case, with exactly five nodes) that maximizes
    the sum of the node scores and edge weights. The greedy method starts from the pair of
    genes with the highest dif_path. To grow the module from this initial pair of genes,
    we then test each other remaining gene to find one, which, together with the existing
    gene set, will create a set with the highest sum of node and edge scores.
    '''
    G = nx.Graph()
    for node, score in node_scores.items():
        G.add_node(node, score=score)
    for edge, weight in edge_weights.items():
        G.add_edge(*edge, weight=weight)

    # Create a list of edges and their weights sorted by weight
    edges_sorted_by_weight = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    # Start with the edge of highest weight
    first_edge = edges_sorted_by_weight[0]
    selected_nodes = list([first_edge[0], first_edge[1]])

    # Create a list of nodes and their scores sorted by score
    nodes_sorted_by_weight = sorted(G.nodes(data=True), key=lambda x: x[1]['score'], reverse=True)

    # Repeat until we have selected the desired number of nodes
    while len(selected_nodes) < num_nodes:
        # Among the remaining nodes, select the node that,
        # when added to the set, maximizes the increase in the sum of node score and edge weight
        max_additional_weight = -float('inf')
        next_node = None
        for node, data in nodes_sorted_by_weight:
            if node in selected_nodes:
                continue
            additional_weight = data['score'] + penalty * sum(G.edges[node, neighbor]['weight'] for neighbor in selected_nodes if (node, neighbor) in G.edges())
            if additional_weight > max_additional_weight:
                max_additional_weight = additional_weight
                next_node = node
        selected_nodes.append(next_node)

    return selected_nodes

def get_genes_in_kegg():
    gene_name = pd.read_table(filepath_or_buffer='./data/kegg/gene_id_convert_by_BioMart.txt',sep=',',header=0,index_col=None)
    gene_name = gene_name.iloc[:,1:3]
    gene_name.columns = ['id','name']
    gene_name.drop_duplicates(subset=['id'],keep='first',inplace=True)
    gene_name = gene_name.dropna(how='any')
    return gene_name['name'].values.tolist()

def get_gene_path_mat():
    pathways = load_obj('./data/kegg/kegg_gene_set.pkl')
    ppnet, ppnodes = load_net_file('./data/network/PPI_cheng_geneName_proced.txt')
    keggnodes = pd.DataFrame({'name': get_genes_in_kegg()})
    nodes = pd.merge(left=keggnodes, right=ppnodes, how='inner', left_on='name', right_on='name')
    nodes.drop_duplicates(keep='first', inplace=True)
    id_df = pd.read_table(filepath_or_buffer='./data/kegg/gene_id_convert_by_BioMart.txt', sep=',', header=0,index_col=None)
    id_df.drop_duplicates(['HGNC symbol'], inplace=True)
    nodes = pd.merge(left=nodes, right=id_df, how='left', left_on='name', right_on='HGNC symbol')
    nodes_ls = nodes['name'].values.tolist()

    gene_path = np.zeros((nodes.shape[0], len(pathways)), dtype=np.int32)
    gene_path = pd.DataFrame(gene_path, index=nodes['name'], columns=list(pathways.keys()))
    for p in pathways.keys():
        p_genes = pathways[p]
        for g in p_genes:
            if g in nodes_ls:
                gene_path.loc[g, p] = 1
    return gene_path