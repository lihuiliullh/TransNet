## Author: Han Wu (han.wu@sydney.edu.au)

from embedding import *
from d_embedding import *
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

# regularize the model and prevent overfitting by randomly dropping entire paths through the network during training.
# used in two transformers
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        residual = x
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask==0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class transformer_block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        print('attn_drop: {}, drop: {}, drop path rate: {}'.format(attn_drop, drop, drop_path))
        self.out_dim = out_dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0.2, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)
        
        #self.deepsets = DeepSets(input_dim=out_dim, hidden_dim=out_dim * 2, output_dim=out_dim)

    def forward(self, x):
        
        size = x.shape # [1024, 5, 100], 1024 batch_size, 5 few-shot, 100 embed_dim
        x, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = self.drop_path(self.mlp(self.norm2(x)))
        
        return x.mean(dim=1).view(size[0], 1, 1, self.out_dim), x.view(-1, self.out_dim)

        #return self.deepsets(x), x.view(-1, self.out_dim)
    
class ContextLearner(nn.Module):
    def __init__(self, num_symbols, embed, embed_dim, few, batch_size,
                 dim, num_heads, qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(ContextLearner, self).__init__()
        self.num_symbols = num_symbols
        self.embed_dim = embed_dim
        self.few = few
        self.batch_size = batch_size
        self.symbol2emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx=self.num_symbols)
        self.symbol2emb.weight.data.copy_(torch.from_numpy(embed))
        self.symbol2emb.weight.requires_grad = True   
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0.2, proj_drop=drop)
        self.fc = nn.Linear(dim, dim//2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, connections, mask):
        relations = connections[:, :, :, 0]
        entities = connections[:, :, :, 1]
        
        rel_embeds = self.symbol2emb(relations)  # Lihui: 1024 x 2 x 100 x 100                  
        entity_embeds = self.symbol2emb(entities)  # Lihui: 1024 x 2 x 100 x 100               
        
        neighbor_embeds = torch.cat((rel_embeds, entity_embeds), dim=3).reshape(-1, 100, self.embed_dim*2) # LIhui: 2048 x 100 x 200
        mask = mask.reshape(-1, 100, 100) # LIhui: 2048 x 100 x 100

        neighbor_embeds, attn = self.attn(self.norm1(neighbor_embeds), mask) # # LIhui: neighbor_embeds 2048 x 100 x 200, attn [2048, 1, 100, 100]
        neighbor_embeds = self.drop_path(neighbor_embeds) 
        # Lihui: given a relation, it has 100 neighbors in total. 50 for head, 50 for tail 
        # calculate simiarity between each neighbor pair, result a 100 x 100 matrix 
        # calculate the mean of the matrix, this is each neighbor's similarity with all other neighors, then calculate the mean, treat as this neighbor's important score 
        # then take the weighted sum of all the neighbor embedding 
        weighted_context = torch.bmm(attn.mean(dim=2), neighbor_embeds.squeeze(1)) # 2048 x 1 x 200
        weighted_context = self.drop_path(self.fc(self.norm2(weighted_context))) # 2048 x 1 x 100
        return weighted_context.squeeze(1)    

# Lihui: this one is the final score function, MTransD
# self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few, transfer_vector)
# class EmbeddingLearner(nn.Module):
#     def __init__(self, input_size):
#         super(EmbeddingLearner, self).__init__()
#         # First layer: input to hidden
#         self.fc1 = nn.Linear(input_size, input_size)
#         # Second layer: hidden to output
#         self.fc2 = nn.Linear(input_size, input_size)
#         # Activation function
#         self.relu = nn.ReLU()

#     def forward(self, h, t, r, pos_num, norm_transfer):
#         # TransD
#         h_transfer, r_transfer, t_transfer = norm_transfer
#         h_transfer = h_transfer[:,:1,:,:]
#         r_transfer = r_transfer[:,:1,:,:]
#         t_transfer = t_transfer[:,:1,:,:]	

#         h_prim = torch.sum(h * h_transfer, -1, True) * r_transfer
#         h_prim = self.relu(self.fc1(h_prim))
#         h_prim = self.fc2(h_prim)
#         h = h + h_prim

#         t_prim = torch.sum(t * t_transfer, -1, True) * r_transfer
#         t_prim = self.relu(self.fc1(t_prim))
#         t_prim = self.fc2(t_prim)
#         t = t + t_prim
        
#         score = -torch.norm(h + r - t, 2, -1).squeeze(2)
#         p_score = score[:, :pos_num]
#         n_score = score[:, pos_num:]
#         return p_score, n_score


############################
class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num, norm_transfer):
        # TransD
        h_transfer, r_transfer, t_transfer = norm_transfer
        h_transfer = h_transfer[:,:1,:,:]
        r_transfer = r_transfer[:,:1,:,:]
        t_transfer = t_transfer[:,:1,:,:]	
        h = h + torch.sum(h * h_transfer, -1, True) * r_transfer
        t = t + torch.sum(t * t_transfer, -1, True) * r_transfer
        
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score

# class EmbeddingLearner(nn.Module):
#     def __init__(self, embed_dim):
#         """
#         EmbeddingLearner module with matrix transformations for h and t.

#         Args:
#             embed_dim (int): Dimension of the embedding vectors.
#         """
#         super(EmbeddingLearner, self).__init__()
#         # Learnable transformation matrices for h and t
#         self.h_transform = nn.Parameter(torch.randn(embed_dim, embed_dim))
#         self.t_transform = nn.Parameter(torch.randn(embed_dim, embed_dim))

#     def forward(self, h, t, r, pos_num, norm_transfer):
#         """
#         Forward pass for embedding scoring with transformations.

#         Args:
#             h (torch.Tensor): Embedding for head entities (batch_size, num_entities, embed_dim).
#             t (torch.Tensor): Embedding for tail entities (batch_size, num_entities, embed_dim).
#             r (torch.Tensor): Embedding for relations (batch_size, num_relations, embed_dim).
#             pos_num (int): Number of positive samples.
#             norm_transfer (tuple): Normalization transfers for h, r, t.

#         Returns:
#             torch.Tensor: Positive scores.
#             torch.Tensor: Negative scores.
#         """
#         h_transfer, r_transfer, t_transfer,  h_transfer2, r_transfer2, t_transfer2 = norm_transfer
#         h_transfer = h_transfer[:,:1,:,:]
#         r_transfer = r_transfer[:,:1,:,:]
#         t_transfer = t_transfer[:,:1,:,:]

#         h_transfer2 = h_transfer2[:,:1,:,:]
#         r_transfer2 = r_transfer2[:,:1,:,:]
#         t_transfer2 = t_transfer2[:,:1,:,:]

#         head1 = torch.sum(h * h_transfer, -1, True) * r_transfer # head1 transform
#         tail1 = torch.sum(t * t_transfer, -1, True) * r_transfer # tail1 transform

#         head2 = torch.sum(h * h_transfer2, -1, True) * r_transfer2 # head2 transform
#         tail2 = torch.sum(t * t_transfer2, -1, True) * r_transfer2 # tail2 transform

#         h = h + head1 + head2
#         t = t + tail1 + tail2

#         score = -torch.norm(h + r - t, 2, -1).squeeze(2)
#         p_score = score[:, :pos_num]
#         n_score = score[:, pos_num:]
#         return p_score, n_score

################# here is the meta clustering ####################
# cluster learning network
# a two-layer neural network with skip-connection
# 𝑞𝑖 = 𝐿𝑒𝑎𝑘𝑦_𝑅𝑒𝐿𝑈 (𝑊𝑞𝑇𝑖)
# 𝑞˜𝑖 = 𝑊𝑝𝑞𝑖 + 𝜇𝑞𝑖
class ClusterPredictor(nn.Module):
    def __init__(self, num_cluster, input_dim, mu=0.2):
        super().__init__()
        self.num_cluster = num_cluster
        
        self.w_q = nn.Linear(input_dim, self.num_cluster) # y=x AT+b.
        self.relu = nn.LeakyReLU(0.1)
        self.w_p = nn.Linear(self.num_cluster, self.num_cluster)
        self.mu = mu
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # input x should be batchsize x embedding dim D
        x = self.w_q(x)
        x = self.relu(x)
        # skip connection
        x = self.w_p(x) + self.mu * x
        return self.softmax(x)
    
class FullyConnect(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, out_features)
        # self.act = act_layer()
        # self.drop1 = nn.Dropout(drop)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop2 = nn.Dropout(drop)

    def forward(self, x, y):
        a = self.fc1(x)
        x = y + a
        return x


    
    
class AttentionUpdateWithSkip(nn.Module):
    def __init__(self, embed_dim):
        """
        Attention-based update with skip connection for preserving 'a'.

        Args:
            embed_dim (int): Dimension of the input vectors.
        """
        super(AttentionUpdateWithSkip, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)  # Transform 'a' vectors (queries)
        self.key_proj = nn.Linear(embed_dim, embed_dim)    # Transform 'b' vectors (keys)
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # Transform 'b' vectors (values)
        self.out_proj = nn.Linear(embed_dim, embed_dim)    # Final projection layer
        self.scale = embed_dim ** 0.5  # Scaling factor for attention

    def forward(self, a, b):
        """
        Forward pass for attention-based vector updates with skip connection.

        Args:
            a (torch.Tensor): Tensor of shape (batch_size, len_a, embed_dim), list of 'a' vectors.
            b (torch.Tensor): Tensor of shape (batch_size, len_b, embed_dim), list of 'b' vectors.

        Returns:
            torch.Tensor: Updated 'a' vectors of shape (batch_size, len_a, embed_dim).
        """
        # Project vectors to query, key, value spaces
        queries = self.query_proj(a)  # Shape: (batch_size, len_a, embed_dim)
        keys = self.key_proj(b)       # Shape: (batch_size, len_b, embed_dim)
        values = self.value_proj(b)   # Shape: (batch_size, len_b, embed_dim)

        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2))  # Shape: (batch_size, len_a, len_b)
        attn_scores = attn_scores / self.scale  # Scale scores
        attn_weights = F.softmax(attn_scores, dim=-1)  # Softmax over 'len_b' dimension

        # Aggregate values from 'b' based on attention weights
        aggregated_info = torch.matmul(attn_weights, values)  # Shape: (batch_size, len_a, embed_dim)

        # Combine aggregated info with the original 'a' using a skip connection
        updated_a = self.out_proj(aggregated_info) + a  # Residual connection

        return updated_a


###########################
class DeepSets(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.g = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Apply φ to each token
        x_phi = self.phi(x)  # Shape: [batch_size, sequence_length, output_dim]
        # Pooling (sum) over the sequence dimension
        pooled = x_phi.sum(dim=1)  # Shape: [batch_size, output_dim]
        # Apply g to the pooled representation
        return self.g(pooled).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, output_dim]


###########################

#############################


class Hire(nn.Module):
    def __init__(self, dataset, parameter, num_symbols, embed=None):
        super(Hire, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.few = parameter['few']
        self.batch_size = parameter['batch_size']
        self.max_neighbor = parameter['max_neighbor']
        self.embedding = Embedding(dataset, parameter) 
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.embedding_learner = EmbeddingLearner()
        self.d_embedding = D_Embedding(dataset, parameter)
        self.d_norm = None
        
        if parameter['dataset'] == 'Wiki-One':
            self.context_learner = ContextLearner(num_symbols, embed, self.embed_dim, self.few, self.batch_size, dim=100, num_heads=1, drop=0.2, drop_path=0.2)
            self.relation_learner = transformer_block(dim=100, out_dim=50, num_heads=1, drop=0.2, drop_path=0.2)
        elif parameter['dataset'] == 'NELL-One':
            self.context_learner = ContextLearner(num_symbols, embed, self.embed_dim, self.few, self.batch_size, dim=200, num_heads=1, drop=0.2, drop_path=0.2)
            self.relation_learner = transformer_block(dim=200, out_dim=100, num_heads=1, drop=0.2, drop_path=0.2)

            relation_embeddings = torch.load('Nell-HiRe_relation_embeddings.pt')
            num_relations = relation_embeddings.size(0)

            #num_relations = 133
            #embedding_dim = relation_embeddings.size(1)
            #self.cluster_embedding = torch.nn.Embedding(num_relations, embedding_dim)
            #self.cluster_embedding.weight.data.copy_(relation_embeddings)

            cluster_K = num_relations
            self.cluster_embedding = nn.Embedding(cluster_K, self.embed_dim)
            

            # assert self.embed_dim == embedding_dim
        else:
            print("Wrong dataset name")
        
        self.finetune = parameter['finetune']
        
        
        ###################################################
        # this is the code for clustering
        # initialize cluster embedding
        # cluster_K = 500
        #cluster_K = 358 # this is for wiki
        # cluster_K = 133 # this for nell
        # self.cluster_embedding = nn.Embedding(cluster_K, self.embed_dim)
        self.cluster_predictor = ClusterPredictor(cluster_K, self.embed_dim)
        self.cluster_FullyConnect = FullyConnect(self.embed_dim + self.embed_dim, out_features=self.embed_dim)
        #self.attentionUpdateWithSkip = AttentionUpdateWithSkip(self.embed_dim)
        ###################################################


    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2
    
    # Lihui: meta is [(left: [1024, 50, 2], degree, right, degree), (left: [1024, 50, 2], degree, right, degree), (left: [1024, 50, 2], degree, right, degree)] when few_shot=3
    def build_context(self, meta): # Lihui: input meta is [few_shot x ]
        left_connections = torch.stack([meta[few_id][0] for few_id in range(self.few)], dim=1)  # Lihui: 1024 x 1 x 50 x 2, size is 1024 x 2 x 50 x 2 when few_shot_num is 2
        right_connections = torch.stack([meta[few_id][2] for few_id in range(self.few)], dim=1) 
        left_degrees = torch.stack([meta[few_id][1] for few_id in range(self.few)], dim=1).reshape(-1)     
        right_degrees = torch.stack([meta[few_id][3] for few_id in range(self.few)], dim=1).reshape(-1)
        
        left_digits = torch.zeros(self.batch_size*self.few, self.max_neighbor).to(self.device)
        right_digits = torch.zeros(self.batch_size*self.few, self.max_neighbor).to(self.device)
        for i in range(self.batch_size*self.few):
            left_digits[i, :left_degrees[i]] = 1
            right_digits[i, :right_degrees[i]] = 1
        left_digits = left_digits.reshape(-1, self.few, self.max_neighbor) # Lihui: This is the neighbor mask 
        right_digits = right_digits.reshape(-1, self.few, self.max_neighbor)
        
        connections = torch.cat((left_connections, right_connections), dim=2) # Lihui: 1024 x 2 x 100 x 2     
        mask = torch.cat((left_digits, right_digits), dim=2) # 1024 x 2 x 100       
        mask_matrix = mask.reshape(-1, self.max_neighbor*2).unsqueeze(2) # 2048 x 100 x 1
        mask = torch.bmm(mask_matrix, mask_matrix.transpose(1,2)) # 2048 x 100 x 100
        
        return connections, mask.reshape(self.batch_size, self.few, self.max_neighbor*2, self.max_neighbor*2)

    # support_meta: Lihui: [(left: [1024, 50, 2], degree, right, degree), (left: [1024, 50, 2], degree, right, degree), (left: [1024, 50, 2], degree, right, degree)] when few_shot=3
    # Lihui: relation embedding learner and context embedding learner use two different embedding systems. Both of them have embedding for nodes and relations.
    # # task: batch_size, few_shot, triplet 
    # This paper to learn relation embedding, simply contact the pretrained (h,t) entity embedding, then use a transformer model to transform the embedding, then take the average
    # My idea, use edge gnn to learn relation embedding. For the relation only in test, use train to learn relation embedding, after learn the gnn model, use it to output edge embedding
    # with all edge embeddings, divide relation embedding to train and test, test will use a model to calulate the similarity and aggregate information. This model is trained on train.
    def forward(self, task, iseval=False, curr_rel='', support_meta=None, support_negative_meta=None):
        support, support_negative, query, negative = [self.embedding(t) for t in task] # [1024, 1, 2, 100] [1024, 1, 2, 100] [1024, 3, 2, 100] [1024, 3, 2, 100] for few_shot_num=1
        # for few_shot_num=2 [1024, 2, 2, 100] [1024, 2, 2, 100] [1024, 3, 2, 100] [1024, 3, 2, 100] 
        transfer_vector = self.d_embedding(task[0]) # Lihui: this is a totally new embedding, this is the parameter of MTransD 
        
        batch_size = support.shape[0]
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative
        
        # positive and negative views
        # Lihui: this is only for contrastive learning 
        if not iseval:
            positive_connections, positive_mask = self.build_context(support_meta)
            negative_connection_mask = [self.build_context(support_nn) for support_nn in support_negative_meta]

            positive_context = self.context_learner(positive_connections, positive_mask) # Lihui: 2048 x 100
            negative_context = [self.context_learner(negative_cm[0], negative_cm[1]) for negative_cm in negative_connection_mask]
        else:
            positive_context, negative_context = None, None
        # Lihui: support size 1024 x 2 x 2 x 100
        # after funtion: rel 1024 x 1 x 1 x 100, support_emb 2048 x 100
        # transformer will take the mean of the output multi-head attention embedding
        rel, support_emb = self.relation_learner(support.contiguous().view(batch_size, few, -1)) 
        
        
        
        # support_emb or rel is the task embeddnig
        ##############################################################
        #if self.finetune:
        sequeeze_rel = torch.squeeze(rel, 1)
        sequeeze_rel = torch.squeeze(sequeeze_rel, 1)

        # # # --- for pretrain
        # # xxxx = sequeeze_rel.clone().detach()
        # #sequeeze_rel = self.attentionUpdateWithSkip(sequeeze_rel, self.cluster_embedding.weight.data)
        
        # # static_rel = sequeeze_rel.clone().detach()
        # # cluster_prob = self.cluster_predictor(static_rel)
        # # c_i = cluster_prob @ self.cluster_embedding.weight.data # c_i should be treated as the input of the model
        # # xxxxx = self.cluster_FullyConnect(torch.cat((static_rel, c_i), 1), static_rel)
        # # --- for pretrain
        
        cluster_prob = self.cluster_predictor(sequeeze_rel)
        c_i = cluster_prob @ self.cluster_embedding.weight.data
        sequeeze_rel = self.cluster_FullyConnect(torch.cat((sequeeze_rel, c_i), 1), sequeeze_rel)
        
        
        rel = torch.unsqueeze(sequeeze_rel, 1)
        rel = torch.unsqueeze(rel, 1)
        ##############################################################
        rel.retain_grad()
        
        transfer_vector[0].retain_grad() # Lihui: h embedding, parameter of MTransD
        transfer_vector[1].retain_grad() # Lihui: r embedding, parameter of MTransD
        transfer_vector[2].retain_grad() # Lihui: t embedding, parameter of MTransD
        # Lihui: embedding learner might need to retrain gradient, wait for next round 

        
        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1) # Lihui: effectively "repeating" the contents of rel along the second dimension without copying the data.

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
            p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few, transfer_vector) # Lihui: embedding_learner is MTransD 
            y = torch.Tensor([1] * batch_size).unsqueeze(dim=1).to(self.device) # Lihui: I revised here 
            self.zero_grad() # Lihui: this is very important
            loss = self.loss_func(p_score, n_score, y)
            loss.backward(retain_graph=True)

            # Lihui: use support graident to update the model parameter
            rel_grad_meta = rel.grad
            rel_q = rel - self.beta * rel_grad_meta
            h_grad_meta = transfer_vector[0].grad
            r_grad_meta = transfer_vector[1].grad
            t_grad_meta = transfer_vector[2].grad
            
            norm_h = transfer_vector[0] - self.beta * h_grad_meta
            norm_r = transfer_vector[1] - self.beta * r_grad_meta
            norm_t = transfer_vector[2] - self.beta * t_grad_meta
            
            norm_transfer = (norm_h, norm_r, norm_t)

            self.rel_q_sharing[curr_rel] = rel_q
            self.d_norm = (transfer_vector[0].mean(0).unsqueeze(0), transfer_vector[1].mean(0).unsqueeze(0), transfer_vector[2].mean(0).unsqueeze(0)) # Lihui: why store mean here??

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative) 
        
        if iseval:
            norm_transfer = self.d_norm
        # Lihui: use query set to calculate the final loss
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, norm_transfer)

        return p_score, n_score, positive_context, negative_context, support_emb

