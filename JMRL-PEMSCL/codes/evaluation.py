import os
import os.path
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from stricts_rules_docred import strict_rules_list

ALLOWED_RELATIONS = strict_rules_list

rel2id = json.load(open('../dataset_docred/meta/rel2id.json', 'r'))
ner2id = json.load(open('../dataset_docred/meta/ner2id.json', 'r'))

id2rel = {value: key for key, value in rel2id.items()}
id2ner = {value: key for key, value in ner2id.items()}

def filter_allowed_relations(predictions, folder, filepath):
    # Load data once
    with open(os.path.join(folder, filepath), 'r') as f:
        data = json.load(f)

    # Create a lookup dictionary for entity types
    entity_types = defaultdict(dict)
    for doc in data:
        title = doc["title"]
        entity_types[title] = {
            i: entity[0]["type"] 
            for i, entity in enumerate(doc["vertexSet"])
        }

    filtered_predictions = []
    filtered_out = []

    for pred in predictions:
        title = pred["title"]
        h_type = entity_types[title].get(pred["h_idx"])
        t_type = entity_types[title].get(pred["t_idx"])
        
        if h_type and t_type:
            rel_tuple = (h_type, str(pred["r"]), t_type)
            if rel_tuple in ALLOWED_RELATIONS:
                filtered_predictions.append(pred)
            else:
                filtered_out.append(rel_tuple)

    print(f"Total predictions before filtering: {len(predictions)}")
    print(f"Total predictions after filtering: {len(filtered_predictions)}")
    print(f"Filtered out relations: {len(filtered_out)}")

    return filtered_predictions


def get_title2pred(pred: list) -> dict:
    '''
    Convert predictions into dictionary.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score']  
    Output:
        :title2pred: dictionary with (key, value) = (title, {rel_triple: score})
    '''
    
    title2pred = {}

    for p in pred:
        if p["r"] == "Na":
            continue
        curr = (p["h_idx"], p["t_idx"], p["r"])
        
        if p["title"] in title2pred:
            if curr in title2pred[p["title"]]:
                title2pred[p["title"]][curr] = max(p["score"], title2pred[p["title"]][curr])
            else:
                title2pred[p["title"]][curr] = p["score"]
        else:
            try:
                title2pred[p["title"]] = {curr: p["score"]}
            except:
                pass
    return title2pred


def get_title2gt(features: dict) -> dict:
    '''
    Convert ground-truth labels to dictionary.
    Input:
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
    Output:
        :title2gt: dictionary with (key, value) = (title, [gold_triples])
    '''
    title2gt = {}
    for f in features:
        title = f["title"]
        title2gt[title] = []
        for idx, p in enumerate(f["hts"]): 
            h,t = p
            label = np.array(f['labels'][idx])
            rs = np.nonzero(label[1:])[0] + 1 # + 1 for no-label
            title2gt[title].extend([(h,t,id2rel[r]) for r in rs])
            
    return title2gt

def select_thresh(cand: list, num_gt: int, correct: int, num_pred: int):
    '''
    select threshold for relation predictions.
    Input:
        :cand: list of relation candidates
        :num_gt: number of ground-truth relations.
        :correct: number of correct relation predictions selected.
        :num_pred: number of relation predictions selected.
    Output:
        :thresh: threshold for selecting relations.
        :sorted_pred: predictions selected from cand. 
    '''
    
    sorted_pred = sorted(cand, key=lambda x:x[1], reverse=True)
    precs, recalls = [], []
    
    for pred in sorted_pred:     
        correct += pred[0]
        num_pred += 1
        precs.append(correct / num_pred)  # Precision
        recalls.append(correct / num_gt)  # Recall                             

    recalls = np.asarray(recalls, dtype='float32')
    precs = np.asarray(precs, dtype='float32')
    f1_arr = (2 * recalls * precs / (recalls + precs + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = sorted_pred[f1_pos][1]

    print('Best thresh', thresh, '\tbest F1', f1)
    return thresh, sorted_pred[:f1_pos + 1]


def merge_results(pred: list, pred_pseudo: list, features: list, thresh: float = None):
    '''
    Merge relation predictions from the original document and psuedo documents.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :pred_pseudo: list of dictionaries, each dictionary entry is a predicted relation triple from pseudo documents. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :thresh: threshold for selecting predictions.
    Output:
        :merged_res: list of merged relation predictions. Each relation prediction is a dictionay with keys (title, h_idx, t_idx, r).
        :thresh: threshold of selecting relation predictions.
    '''
    
    title2pred = get_title2pred(pred)
    title2pred_pseudo = get_title2pred(pred_pseudo)

    title2gt = get_title2gt(features)
    num_gt = sum([len(title2gt[t]) for t in title2gt])

    titles = list(title2pred.keys())
    cand = []
    merged_res = []
    correct, num_pred = 0, 0

    for t in titles:
        rels = title2pred[t]
        rels_pseudo = title2pred_pseudo[t] if t in title2pred_pseudo else {}

        union = set(rels.keys()) | set(rels_pseudo.keys())
        for r in union:
            if r in rels and r in rels_pseudo: # add those into predictions
                if rels[r] > 0 and rels_pseudo[r] > 0:
                    merged_res.append({'title':t, 'h_idx':r[0], 't_idx':r[1], 'r': r[2]})
                    num_pred += 1
                    correct += r in title2gt[t]
                    continue
                score = rels[r] + rels_pseudo[r]
            elif r in rels: # -10 for penalty
                score = rels[r] - 10
            elif r in rels_pseudo:
                score = rels_pseudo[r] - 10
            cand.append((r in title2gt[t], score, t, r[0], r[1], r[2]))
    
    if thresh != None:
        sorted_pred = sorted(cand, key=lambda x:x[1], reverse=True)
        last = min(filter(lambda x: x[1] > thresh, sorted_pred))
        until = sorted_pred.index(last)
        cand = sorted_pred[:until + 1]
        merged_res.extend([{'title':r[2], 'h_idx':r[3], 't_idx':r[4], 'r': r[5]} for r in cand])
        return merged_res, thresh

    if cand != []:
        thresh, cand = select_thresh(cand, num_gt, correct, num_pred)
        merged_res.extend([{'title':r[2], 'h_idx':r[3], 't_idx':r[4], 'r': r[5]} for r in cand])

    return merged_res, thresh


def extract_relative_score(scores: list, topks: list) -> list:
    '''
    Get relative score from topk predictions.
    Input:
        :scores: a list containing scores of topk predictions.
        :topks: a list containing relation labels of topk predictions.
    Output:
        :scores: a list containing relative scores of topk predictions.
    '''
    
    na_score = scores[-1].item() - 1
    if 0 in topks:
        na_score = scores[np.where(topks==0)].item()     
    
    scores -= na_score

    return scores


def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res

def gen_train_facts(data_file_name, truth_dir):
    
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train



def calculate_metrics(dict_classe):
    precision = dict_classe["tp"] / (dict_classe["tp"] + dict_classe["fp"]) 
    recall = dict_classe["tp"] / (dict_classe["tp"] + dict_classe["fn"]) 
    f1_score = (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1_score": f1_score}


def calculate_f1_score_weighted(classes_dict):
    # print(classes_dict)
    return sum([(classes_dict[i]["gold_relations"] * classes_dict[i]["f1"] ) for i in classes_dict.keys()]) / sum([classes_dict[i]["gold_relations"] for i in classes_dict.keys()])

def official_evaluate(tmp, path, train_file = "train_annotated.json", dev_file = None):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = [] # gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))
    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])
    # print("len sub :",len(submission_answer))
    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1


    relation_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
    total_entity_pairs = 0
    for x in truth:
        n_entities = len(x['vertexSet'])
        total_entity_pairs += n_entities * (n_entities - 1) // 2

    total_entity_pairs = sum(len(x['vertexSet']) * (len(x['vertexSet']) - 1) for x in truth)

    for r in rel2id.keys():
        relation_metrics[r]['tn'] = total_entity_pairs

    for x in submission_answer:
        title, r, h_idx, t_idx = x['title'], x['r'], x['h_idx'], x['t_idx']
        if (title, r, h_idx, t_idx) in std:
            relation_metrics[r]['tp'] += 1
        else:
            relation_metrics[r]['fp'] += 1

    for (title, r, h_idx, t_idx) in std:
        if not any(x['title'] == title and x['r'] == r and x['h_idx'] == h_idx and x['t_idx'] == t_idx for x in submission_answer):
            relation_metrics[r]['fn'] += 1

    for r in relation_metrics:
        relation_metrics[r]['tn'] -= (relation_metrics[r]['tp'] + relation_metrics[r]['fp'] + relation_metrics[r]['fn'])

    
    num_relations = len(relation_metrics)
    fps = 0
    fns = 0
    re_p_macro, re_r_macro, re_f1_macro = 0, 0, 0
    print(relation_metrics)
    for r in relation_metrics:
        tp = relation_metrics[r]['tp']
        fp = relation_metrics[r]['fp']
        fn = relation_metrics[r]['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        re_p_macro += p
        re_r_macro += r
        re_f1_macro += f1
        fns += fn
        fps += fp
    re_p_macro /= num_relations if num_relations > 0 else 1
    re_r_macro /= num_relations if num_relations > 0 else 1
    re_f1_macro /= num_relations if num_relations > 0 else 1

        

    re_f1_micro = correct_re / (correct_re + 0.5 *(fps +fns))

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)


    return {
        'relation_extraction': {
            'precision': re_p_macro,
            'recall': re_r_macro,
            'f1_micro': re_f1_micro,
            'f1_macro': re_f1_macro,
            'f1_ignore_train_annotated': re_f1_ignore_train_annotated,
            'f1_ignore_train': re_f1_ignore_train
        },
    
        'evidence':{
            'precision': evi_p,
            'recall': evi_r,
            'f1_micro': evi_r,
        }
    }