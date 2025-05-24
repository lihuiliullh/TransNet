import random
import numpy as np


class DataLoader(object):
    def __init__(self, dataset, parameter, step='train'):
        self.curr_rel_idx = 0
        self.tasks = dataset[step+'_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.e1rel_e2 = dataset['e1rel_e2']
        self.all_rels = sorted(list(self.tasks.keys()))
        self.num_rels = len(self.all_rels)
        self.few = parameter['few']
        self.bs = parameter['batch_size']
        self.nq = parameter['num_query']

        if step != 'train':
            self.eval_triples = []
            for rel in self.all_rels:
                self.eval_triples.extend(self.tasks[rel][self.few:])
            self.num_tris = len(self.eval_triples)
            self.curr_tri_idx = 0

    def next_one(self):
        # shift curr_rel_idx to 0 after one circle of all relations
        if self.curr_rel_idx % self.num_rels == 0:
            random.shuffle(self.all_rels)
            self.curr_rel_idx = 0

        # get current relation and current candidates
        curr_rel = self.all_rels[self.curr_rel_idx] # Lihui: all_rels stores all relations in "train" or "test" or "dev", not and
        self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels  # shift current relation idx to next
        curr_cand = self.rel2candidates[curr_rel]  # candidate find by GMatching
        while len(curr_cand) <= 10 or len(self.tasks[curr_rel]) <= 10:  # ignore the small task sets
            curr_rel = self.all_rels[self.curr_rel_idx] # Lihui: they focus on few shot relation in range 30 to 50
            self.curr_rel_idx = (self.curr_rel_idx + 1) % self.num_rels
            curr_cand = self.rel2candidates[curr_rel]

        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = self.tasks[curr_rel] # Lihui: self.tasks[curr_rel] is a triplet list of relation curr_rel 
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.few+self.nq) # Lihui: self.nq is the number of query
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.few]]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.few:]]

        # construct support and query negative triples
        support_negative_triples = [] # Lihui: each support triplet has a negative sample 
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand) # Lihui: random choose one from hard negative sample 
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2: # Lihui: not in the neighbor, and not equal to current tail 
                    break
            support_negative_triples.append([e1, rel, negative])

        negative_triples = [] # Lihui: each query triplet has a negative sample 
        for triple in query_triples:
            e1, rel, e2 = triple
            while True:
                negative = random.choice(curr_cand)  # Lihui: random choose one from hard negative sample 
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
            negative_triples.append([e1, rel, negative])

        return support_triples, support_negative_triples, query_triples, negative_triples, curr_rel

    # call this one when training 
    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.bs)] # Lihui: self.bs is the batch size
        # Lihui: output like this
        # [
        #     (support1, support_neg1, query1, neg1, rel1),
        #     (support2, support_neg2, query2, neg2, rel2),
        #     (support3, support_neg3, query3, neg3, rel3)
        # ]

        support, support_negative, query, negative, curr_rel = zip(*next_batch_all)
        # Lihui: after zip, the data will look like
        # (
        #     (support1, support2, support3),
        #     (support_neg1, support_neg2, support_neg3),
        #     (query1, query2, query3),
        #     (neg1, neg2, neg3),
        #     (rel1, rel2, rel3)
        # )
        # 
        # One example output 
        # (
        #     [
        #         # 支持、负采样支持、查询、负采样查询三元组
        #         [
        #             # 支持三元组
        #             [[('e1a', 'rel1', 'e2a'), ('e1b', 'rel1', 'e2b')],
        #              [('e1c', 'rel2', 'e2c'), ('e1d', 'rel2', 'e2d')],
        #              [('e1e', 'rel3', 'e2e'), ('e1f', 'rel3', 'e2f')]],
                    
        #             # 负采样支持三元组
        #             [[('e1a', 'rel1', 'neg1'), ('e1b', 'rel1', 'neg2')],
        #              [('e1c', 'rel2', 'neg3'), ('e1d', 'rel2', 'neg4')],
        #              [('e1e', 'rel3', 'neg5'), ('e1f', 'rel3', 'neg6')]],

        #             # 查询三元组
        #             [[('e1g', 'rel1', 'e2g'), ('e1h', 'rel1', 'e2h')],
        #              [('e1i', 'rel2', 'e2i'), ('e1j', 'rel2', 'e2j')],
        #              [('e1k', 'rel3', 'e2k'), ('e1l', 'rel3', 'e2l')]],

        #             # 负采样查询三元组
        #             [[('e1g', 'rel1', 'neg7'), ('e1h', 'rel1', 'neg8')],
        #              [('e1i', 'rel2', 'neg9'), ('e1j', 'rel2', 'neg10')],
        #              [('e1k', 'rel3', 'neg11'), ('e1l', 'rel3', 'neg12')]]
        #         ]
        #     ],

        #     # 当前关系
        #     ('rel1', 'rel2', 'rel3')
        # )
        return [support, support_negative, query, negative], curr_rel

    def next_one_on_eval(self):
        if self.curr_tri_idx == self.num_tris: # Lihui: the evaluation is over, return "EOT"
            return "EOT", "EOT"

        # get current triple
        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel] # Lihui: all triplets of current relation 

        # get support triples
        support_triples = curr_task[:self.few] # Lihui: select support triplets 

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples: # Lihui: for each triplet in the support, get an negative triplet
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2: # Lihui: this is hard negative samples 
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2: # Lihui: generate a lot of negatives 
                negative_triples.append([e1, rel, negative])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel

    def next_one_on_eval_by_relation(self, curr_rel):
        if self.curr_tri_idx == len(self.tasks[curr_rel][self.few:]):
            self.curr_tri_idx = 0
            return "EOT", "EOT"

        # get current triple
        query_triple = self.tasks[curr_rel][self.few:][self.curr_tri_idx]
        self.curr_tri_idx += 1
        # curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        # get support triples
        support_triples = curr_task[:self.few]

        # construct support negative
        support_negative_triples = []
        shift = 0
        for triple in support_triples:
            e1, rel, e2 = triple
            while True:
                negative = curr_cand[shift]
                if (negative not in self.e1rel_e2[e1 + rel]) \
                        and negative != e2:
                    break
                else:
                    shift += 1
            support_negative_triples.append([e1, rel, negative])

        # construct negative triples
        negative_triples = []
        e1, rel, e2 = query_triple
        for negative in curr_cand:
            if (negative not in self.e1rel_e2[e1 + rel]) \
                    and negative != e2:
                negative_triples.append([e1, rel, negative])

        support_triples = [support_triples]
        support_negative_triples = [support_negative_triples]
        query_triple = [[query_triple]]
        negative_triples = [negative_triples]

        return [support_triples, support_negative_triples, query_triple, negative_triples], curr_rel
