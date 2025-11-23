# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv

from lavila.models.tokenizer import MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer


def generate_label_map(dataset, args):
    if dataset == 'ek100_cls':
        finetune_type = getattr(args, 'egtea_finetune_type', 'action')  # Use same arg name for consistency
        train_csv = getattr(
            args, 'ek100_train_csv',
            '/home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_train.csv'
        )
        val_csv = getattr(
            args, 'ek100_val_csv',
            '/home/dz/Projects/multi-modal_AR/data/EK/data/EPIC_100_validation.csv'
        )
        
        if finetune_type == 'verb':
            print("Preprocess ek100 verb label space")
            verb_list = []
            mapping_verb2narration = {}
            for f in [train_csv, val_csv]:
                csv_reader = csv.reader(open(f))
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    verb_class = int(row[10])
                    verb_name = row[9]
                    if verb_class not in verb_list:
                        verb_list.append(verb_class)
                    if verb_class not in mapping_verb2narration:
                        mapping_verb2narration[verb_class] = []
                    mapping_verb2narration[verb_class].append(verb_name)
            
            verb_list = sorted(verb_list)
            print('# of verbs = {}'.format(len(verb_list)))
            mapping_vn2act = {str(v): i for i, v in enumerate(verb_list)}
            labels = [list(set(mapping_verb2narration[verb_list[i]])) for i in range(len(mapping_vn2act))]
            print('First 5 verb labels:', labels[:5])
            
        elif finetune_type == 'noun':
            print("Preprocess ek100 noun label space")
            noun_list = []
            mapping_noun2narration = {}
            for f in [train_csv, val_csv]:
                csv_reader = csv.reader(open(f))
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    noun_class = int(row[12])
                    noun_name = row[11]
                    if noun_class not in noun_list:
                        noun_list.append(noun_class)
                    if noun_class not in mapping_noun2narration:
                        mapping_noun2narration[noun_class] = []
                    mapping_noun2narration[noun_class].append(noun_name)
            
            noun_list = sorted(noun_list)
            print('# of nouns = {}'.format(len(noun_list)))
            mapping_vn2act = {str(n): i for i, n in enumerate(noun_list)}
            labels = [list(set(mapping_noun2narration[noun_list[i]])) for i in range(len(mapping_vn2act))]
            print('First 5 noun labels:', labels[:5])
            
        else:  # action (default)
            print("Preprocess ek100 action label space")
            vn_list = []
            mapping_vn2narration = {}
            for f in [train_csv, val_csv]:
                csv_reader = csv.reader(open(f))
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    vn = '{}:{}'.format(int(row[10]), int(row[12]))
                    narration = row[8]
                    if vn not in vn_list:
                        vn_list.append(vn)
                    if vn not in mapping_vn2narration:
                        mapping_vn2narration[vn] = [narration]
                    else:
                        mapping_vn2narration[vn].append(narration)
                    # mapping_vn2narration[vn] = [narration]
            vn_list = sorted(vn_list)
            print('# of action= {}'.format(len(vn_list)))
            mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
            labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
            print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        class_list_path = getattr(
            args, 'charades_classlist',
            'datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt'
        )
        with open(class_list_path) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        if args.egtea_finetune_type == 'action':
            txt_file = 'action_idx'
        if args.egtea_finetune_type == 'verb':
            txt_file = 'verb_idx'  
        if args.egtea_finetune_type == 'noun':
            txt_file = 'noun_idx'
        idx_root = getattr(
            args, 'egtea_idx_root',
            '../data/EGTEA/raw/annotation/idx'
        )
        idx_path = os.path.join(idx_root, txt_file + '.txt')
        with open(idx_path) as f: # action_idx 106 verb_idx 19 noun_idx 53
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act


def generate_tokenizer(model):
    if model.endswith('DISTILBERT_BASE'):
        tokenizer = MyDistilBertTokenizer('distilbert-base-uncased')
    elif model.endswith('BERT_BASE'):
        tokenizer = MyBertTokenizer('bert-base-uncased')
    elif model.endswith('BERT_LARGE'):
        tokenizer = MyBertTokenizer('bert-large-uncased')
    elif model.endswith('GPT2'):
        tokenizer = MyGPT2Tokenizer('gpt2', add_bos=True)
    elif model.endswith('GPT2_MEDIUM'):
        tokenizer = MyGPT2Tokenizer('gpt2-medium', add_bos=True)
    elif model.endswith('GPT2_LARGE'):
        tokenizer = MyGPT2Tokenizer('gpt2-large', add_bos=True)
    elif model.endswith('GPT2_XL'):
        tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
    else:
        print("Using SimpleTokenizer because of model '{}'. "
              "Please check if this is what you want".format(model))
        tokenizer = SimpleTokenizer()
    return tokenizer
