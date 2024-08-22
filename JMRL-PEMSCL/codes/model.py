import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

from long_seq import process_long_input
from losses import *
from utils import set_seed
from stricts_rules_docred import strict_rules_list


class DocREModel(nn.Module):
    def __init__(self, config, args, model, emb_size=1024, block_size=64, num_labels=-1, T=2, L=20):
        super().__init__()
        set_seed(args)
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.lambda_3 = args.lambda_3
        self.loss_fnt = PMTEMloss(args.lambda_1, args.lambda_2)
        self.SCL_loss = MLLTRSCLloss(tau=args.tau, tau_base=args.tau_base)

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size*block_size, config.num_labels)
        print("nb classes:", config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.T = T
        self.L = L
        self.n = config.num_labels

        self.diff_w = nn.Parameter(torch.Tensor(self.n, self.T, self.L, 2 * self.n + 1))
        nn.init.kaiming_uniform_(self.diff_w.view(self.n, -1), a=np.sqrt(5))
        self.diff_weights = nn.Parameter(torch.Tensor(self.n, self.L, 1))
        nn.init.kaiming_uniform_(self.diff_weights.view(self.n, -1), a=np.sqrt(5))
        strict_rules = strict_rules_list

        self.strict_rules = strict_rules if strict_rules is not None else []
        self.entity_type_to_id = json.load(open(args.ner2id, "r"))
        self.relation_type_to_id = json.load(open(args.rel2id, "r"))  # Dictionnaire mappant les types de relations à des IDs
        
        # Créez une matrice de règles strictes pour un accès rapide
        self.strict_rules_matrix = self.create_strict_rules_matrix()



    def create_strict_rules_matrix(self):
        num_entity_types = len(self.entity_type_to_id)
        num_relation_types = len(self.relation_type_to_id)
        
        # La matrice a maintenant 3 dimensions
        matrix = torch.full((num_entity_types, num_relation_types, num_entity_types), fill_value=-1, dtype=torch.long)
        
        for rule in self.strict_rules:
            e1, r1, e2 = rule
            e1_id = self.entity_type_to_id[e1]
            r1_id = self.relation_type_to_id[r1]
            e2_id = self.entity_type_to_id[e2]            
            matrix[e1_id, r1_id, e2_id] = 1
        return matrix

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        elif config.transformer_type == "camembert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention


    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta", "camembert"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                e_emb, e_att = [], []
                for start, end in e:
                    if start + offset < c:
                        e_emb.append(sequence_output[i, start + offset])
                        e_att.append(attention[i, :, start + offset])
                if len(e_emb) > 0:
                    e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                    e_att = torch.stack(e_att, dim=0).mean(0)
                else:
                    e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def reasoning_by_soft_rules(self, logits):
        n_e = logits.shape[0]
        eye = torch.eye(n_e).to(logits.device)
        input = logits[:, :, :]
        input = torch.cat([input, torch.permute(input, (1, 0, 2)), torch.unsqueeze(eye, dim=-1)], dim=-1)
        all_states = []
        for r in range(self.n):
            cur_states = []
            for t in range(self.T + 1):
                if t == 0:
                    w = self.diff_w[r][t]
                    one_hot = torch.zeros_like(w.detach())
                    if r != 0:
                        one_hot[:, 0] = -1e30
                        one_hot[:, self.n] = -1e30
                    w = torch.softmax(w + one_hot, dim=-1)
                    input_cur = input.view(-1, 2 * self.n + 1)
                    s_tmp = torch.mm(input_cur, torch.permute(w, (1, 0))).view(n_e, -1, self.L)
                    s = s_tmp
                    cur_states.append(s)
                if t >= 1 and t < self.T:
                    w = self.diff_w[r][t]
                    one_hot = torch.zeros_like(w.detach())
                    if r != 0:
                        one_hot[:, 0] = -1e30
                        one_hot[:, self.n] = -1e30
                    w = torch.softmax(w + one_hot, dim=-1)
                    input_cur = torch.permute(input, (0, 2, 1)).reshape(-1, n_e)
                    s_tmp = torch.mm(input_cur, cur_states[t - 1].reshape(n_e, -1))
                    s_tmp = s_tmp.view(n_e, 2 * self.n + 1, -1, self.L)
                    s_tmp = s_tmp.float()
                    w = w.float()
                    s_tmp = torch.einsum('mrnl,lr->mnl', s_tmp, w)
                    s = s_tmp
                    cur_states.append(s)
                if t == self.T:
                    weight = torch.tanh(self.diff_weights[r])
                    final_state = torch.einsum('mnl,lk->mnk', cur_states[-1], weight).squeeze(dim=-1)
                    all_states.append(final_state)
        output = torch.stack(all_states, dim=-1)
        return output

    def apply_strict_rules(self, logits):
        n_e = logits.shape[0]
        strict_output = logits.clone()
        
        # Get the predicted relations for all pairs at once
        predicted_relations = torch.argmax(logits, dim=-1)
        
        # Create a mask for pairs where strict rules apply
        strict_rule_mask = torch.zeros_like(predicted_relations, dtype=torch.bool)
        for i in range(n_e):
            for j in range(n_e):
                rel = predicted_relations[i, j]
                if torch.any(self.strict_rules_matrix[:, rel, :] == 1):
                    strict_rule_mask[i, j] = True
        
        # Apply strict rules where the mask is True
        strict_output[strict_rule_mask] = torch.zeros_like(strict_output[strict_rule_mask])
        strict_output[strict_rule_mask, predicted_relations[strict_rule_mask]] = 1.0
        
        # Set to zero where no strict rule applies
        strict_output[~strict_rule_mask, predicted_relations[~strict_rule_mask]] = 0.0
        
        return strict_output
    
    # Priorité aux règles stricts
    def combine_soft_and_strict_rules(self, soft_output, strict_output):
        combined = torch.where(strict_output > 0, strict_output, soft_output)
        return combined

    # Moyenne pondérée
    # def combine_soft_and_strict_rules(self, soft_output, strict_output, alpha=0.5):
    #     return alpha * soft_output + (1 - alpha) * strict_output
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        logits_rule_soft = []
        start = 0
        for b in range(len(hts)):
            indices = torch.LongTensor(hts[b]).transpose(1, 0).to(logits)
            n_e = int(np.sqrt(len(hts[b]))) + 1
            end = start + len(hts[b])
            input = torch.softmax(logits[start: end, :], dim=-1)
            matrix = torch.sparse.FloatTensor(indices.long(), input,
                                              torch.Size([n_e, n_e, self.config.num_labels])).to(logits)
            logits_rule = self.reasoning_by_soft_rules(matrix.to_dense())
            logits_rule = logits_rule.view(-1, self.config.num_labels)
            indices = indices[0] * n_e + indices[1]
            logits_rule = logits_rule[indices.long()]
            logits_rule_soft.append(logits_rule)
            start = end
        logits_rule_soft_tmp = torch.cat(logits_rule_soft, dim=0)
        logits_rule_soft = logits_rule_soft_tmp + logits

        
        output = (self.loss_fnt.get_label(logits_rule_soft, num_labels=self.num_labels), logits_rule_soft, bl)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)

            loss = self.loss_fnt(logits.float(), labels.float())

            scl_loss = self.SCL_loss(F.normalize(bl,dim=-1), labels)
            loss_cls = loss + scl_loss * self.lambda_3
            loss_rule = self.loss_fnt(logits_rule_soft.float() / 0.2, labels.clone().float())
            loss = loss_cls + loss_rule
            loss_dict = {'loss_cls': loss_cls.item(), 'loss_rule': loss_rule.item()}
            output = (loss.to(sequence_output),) + output + (loss_rule.to(sequence_output),)
        return output

    def forward_combined(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        logits_rule_soft = []
        logits_rule_strict = []
        start = 0
        for b in range(len(hts)):
            indices = torch.LongTensor(hts[b]).transpose(1, 0).to(logits)
            n_e = int(np.sqrt(len(hts[b]))) + 1
            end = start + len(hts[b])
            input = torch.softmax(logits[start: end, :], dim=-1)
            matrix = torch.sparse.FloatTensor(indices.long(), input,
                                            torch.Size([n_e, n_e, self.config.num_labels])).to(logits)
            
            # Application des règles souples
            logits_rule = self.reasoning_by_soft_rules(matrix.to_dense())
            logits_rule = logits_rule.view(-1, self.config.num_labels)
            indices = indices[0] * n_e + indices[1]
            logits_rule = logits_rule[indices.long()]
            logits_rule_soft.append(logits_rule)

            # Application des règles strictes
            logits_strict = self.apply_strict_rules(matrix.to_dense())
            logits_strict = logits_strict.view(-1, self.config.num_labels)
            logits_strict = logits_strict[indices.long()]
            logits_rule_strict.append(logits_strict)

            start = end

        logits_rule_soft = torch.cat(logits_rule_soft, dim=0)
        logits_rule_strict = torch.cat(logits_rule_strict, dim=0)

        # Combinaison des logits originaux, des règles souples et des règles strictes
        combined_logits = self.combine_all_logits(logits, logits_rule_soft, logits_rule_strict)

        output = (self.loss_fnt.get_label(combined_logits, num_labels=self.num_labels), combined_logits, bl)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            scl_loss = self.SCL_loss(F.normalize(bl,dim=-1), labels)
            loss_cls = loss + scl_loss * self.lambda_3
            loss_rule = self.loss_fnt(combined_logits.float() / 0.2, labels.clone().float())
            loss = loss_cls + loss_rule
            loss_dict = {'loss_cls': loss_cls.item(), 'loss_rule': loss_rule.item()}
        
            output = (loss.to(sequence_output),) + output + (loss_rule.to(sequence_output),)
        return output
    
    def combine_all_logits(self, original_logits, soft_rule_logits, strict_rule_logits):
        # Vous pouvez ajuster cette méthode selon vos besoins spécifiques
        alpha = 0.4  # Poids pour les logits originaux
        beta = 0.3   # Poids pour les règles souples
        gamma = 0.3  # Poids pour les règles strictes
        
        combined = alpha * original_logits + beta * soft_rule_logits + gamma * strict_rule_logits
        return combined
    