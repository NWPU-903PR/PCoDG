from torchppr.utils import sparse_normalize
from myutils import *
from tqdm import tqdm as tqdm
import warnings
warnings.filterwarnings("ignore")


#********************************** Step 1 ***************************************************
# Loading and preprocessing mutation, gene expression, and the PPI network from files
#*********************************************************************************************
cancer = 'BRCA'
mut,rnaseq,ppnet_sub,ppnodes_sub=load_omics_data(cancer)

#********************************** Step 2 ***************************************************
# Constructing the personalized hypergraph and scoring the personalized mutated genes
# by using the hypergraph random walk based on hyperedge similarity.
#*********************************************************************************************

# Calculate the outlying matrix (i.e., the DEGs for all patients in a specific cancer dataset)
std_threshold=1
outlier_mat = cal_outlying_gene_matrix(df=rnaseq,std_threshold=std_threshold)

# Applying RWR to measure the impact of personalized mutated genes within the PPI network for each patient.
threshold = 0.001
alpha = 0.3
count_num = 0
mut_rank_dic = dict()
for samp_id in mut.columns.values.tolist():
    if mut[samp_id].sum() > 300 or mut[samp_id].sum() < 5:
        continue
    adj_weight = convert_edge_to_adj(edge_df=ppnet_sub,nodes=ppnodes_sub,weighted=False).astype(np.float32)
    adj_weight = torch.from_numpy(adj_weight).to_sparse()
    adj_normalized = sparse_normalize(adj_weight, dim=0)

    samp_mut = mut[samp_id][mut[samp_id]==1].index.values.tolist() # Mutated gene set for a patient.
    samp_mut_target_dic = dict() # record the impact of each mutated gene in every patient in a dataset.
    time_start = time.time()
    # Applying RWR to measure the impact of each personalized mutated genes within the PPI network
    for i in range(0,len(samp_mut)):
        mg = samp_mut[i]
        mg_idx = ppnodes_sub['name'].values.tolist().index(mg)
        # RWR process of the mutated gene mg_idx
        samp_mg_net = subnet_of_rwr(nodes=ppnodes_sub,adj_normalized=adj_normalized,seeds=[mg_idx],threshold=threshold,alpha=alpha)
        # Extract personalized DEGs from samp_mg_net
        samp_outlier = outlier_mat.loc[:,samp_id]
        samp_outlier = samp_outlier[samp_outlier==1].index.values.tolist()
        samp_mg_outlier = list(set(samp_outlier) & set(samp_mg_net['name'].values.tolist()))
        if mg not in samp_mg_outlier:
            samp_mg_outlier.append(mg)
        samp_mut_target_dic[mg] = samp_mg_outlier
    # Construct the incidence matrix of a personalized hypergraph.
    inc_mtx = create_incidence_matrix(samp_mut_target_dic, mut_in_targets=False)
    # Score the personalized mutated genes by using the hypergraph random walk based on hyperedge similarity
    _, mut_rank_dic[samp_id] = HeRW_on_individual_hypergraph(inc_mtx)
    count_num = count_num + 1
    print('Number: %d, Sample: %s, costing time:%fs' % (count_num,samp_id, time.time() - time_start))

#********************************** Step 3 ***************************************************
# Combining the scores of mutated genes with biological signaling pathway data
# to identify personalized cooperating cancer driver genes using a greedy algorithm.
#*********************************************************************************************
# Load the signaling pathway data
gene_path = get_gene_path_mat() # the incidence matrix of gene and KEGG signaling pathways

synerg_mut_set_dic = dict()
for samp_id in tqdm(mut_rank_dic.keys()):
    s_mut_rank = mut_rank_dic[samp_id]
    s_mut = s_mut_rank.index.values.tolist()
    s_mut_score = {}
    s_mut_difpath = {}
    for m in s_mut:
        s_mut_score[m] = s_mut_rank[m]
    for i in range(0,len(s_mut)):
        for j in range(i+1,len(s_mut)):
            s_mut_difpath[(s_mut[i],s_mut[j])] = dif_path_frequence(gene_path,s_mut[i],s_mut[j])

    s_mut_score_df = pd.DataFrame(s_mut_score,index=['score']).transpose()
    s_mut_difpath_df = pd.DataFrame(s_mut_difpath,index=['difpath']).transpose()

    if (s_mut_score_df.max() - s_mut_score_df.min())[0] > 0:
        s_mut_score_df = (s_mut_score_df - s_mut_score_df.min()) / (s_mut_score_df.max() - s_mut_score_df.min())
    else:
        s_mut_score_df['score'] = [0 for _ in range(0,s_mut_score_df.shape[0])]
    if (s_mut_difpath_df.max() - s_mut_difpath_df.min())[0] > 0:
        s_mut_difpath_df = (s_mut_difpath_df - s_mut_difpath_df.min()) / (s_mut_difpath_df.max() - s_mut_difpath_df.min())
    else:
        s_mut_difpath_df['difpath'] = [0 for _ in range(0,s_mut_difpath_df.shape[0])]
    s_mut_difpath_df.sort_values('difpath', ascending=False, inplace=True)

    s_mut_score_norm = s_mut_score_df.to_dict()['score']
    s_mut_difpath_norm = s_mut_difpath_df.to_dict()['difpath']

    synerg_mut_set_dic[samp_id] = find_heavy_subgraph(node_scores=s_mut_score_norm,edge_weights=s_mut_difpath_norm,penalty=1.0,num_nodes=5)

save_obj(obj=synerg_mut_set_dic,name='./results/CoDGs_%s.pkl'%(cancer))



