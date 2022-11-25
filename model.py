# import keras.backend as k
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertModel, BertForMaskedLM, RobertaModel, RobertaTokenizer, XLNetModel, XLNetTokenizer
from utils import convert_examples_to_features
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_transformers import AdamW, WarmupLinearSchedule, ConstantLRSchedule
from tqdm import tqdm, trange
import random
import numpy as np
from torch.nn import CrossEntropyLoss
import os
from utils import compute_metrics


# # set models' parameters
bert_params = {
    'cls_pos': 0,
    'learning_rate': 5e-5,
    'model_class': BertModel,
    'tokenizer_class': BertTokenizer,
    'pretrained_model_name': 'bert-base-uncased',
    'pretrained_file_path': 'bert-base-uncased',
    'output_hidden_states': True
}



roberta_params = {
    'cls_pos': 0,
    'learning_rate': 1e-5,
    'model_class': RobertaModel,
    'tokenizer_class': RobertaTokenizer,
    'pretrained_model_name': 'roberta-base',
    'pretrained_file_path': './',
    'output_hidden_states': True
}

xlnet_params = {
    'cls_pos': -1,
    'learning_rate': 2e-5,
    'model_class': XLNetModel,
    'tokenizer_class': XLNetTokenizer,
    'pretrained_model_name': 'xlnet-base-cased',
    'pretrained_file_path': './',
    'output_hidden_states': True
}



class Classifier(nn.Module):
    def __init__(self, num_labels, **kwargs):
        """Initialize the components of the classifier."""
        super(Classifier, self).__init__()
        self.cls_pos = kwargs['cls_pos'] 
        self.num_labels = num_labels
        self.model = kwargs['model_class'].from_pretrained(kwargs['pretrained_file_path'], output_hidden_states=True)
        self.tokenizer =  kwargs['tokenizer_class'].from_pretrained(kwargs['pretrained_file_path'])
        
        self.dense = nn.Linear(in_features=768, out_features=768, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(in_features=768, out_features=num_labels, bias=True)
        

    def forward(self, input_ids=None, attention_mask=None, segment_ids = None, labels=None):
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids= segment_ids)#, output_hidden_states=True)
        
        output = torch.tanh(self.dense(output[1]))
        output = self.dropout(output)
        logits = self.out_proj(output)
        last_hidden_states = output[0]
        #hidden_states = output[2]
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))            
            return loss, logits, output, last_hidden_states
        
        return logits, output, last_hidden_states

    def clone_for_refitting(self):
        """
        Create a copy of the classifier that can be refit from scratch. Will inherit same architecture, optimizer and
        initialization as cloned model, but without weights.
        :return: new estimator
        """

#         import tensorflow as tf  # lgtm [py/repeated-import]
#         import keras  # lgtm [py/repeated-import]

        model_clone =  copy.deepcopy(self)
        reset_parameters = getattr(model_clone, "reset_parameters", None)
        if callable(reset_parameters):
            model_clone.reset_parameters()
            
        # with torch.no_grad():
        #     model_clone.weight.fill_(1.)
        return model_clone


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)
        
def train_model(args, X, y, model, tokenizer, device, prefix=""):
    """ Train the model """
    train_features = convert_examples_to_features(X, y, args.max_seq_length, tokenizer)
    
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    
    train_dataset = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_id)
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps =1e-08)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_epochs, t_total=t_total)
                                                     
    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.epochs)
    print("  Instantaneous batch size per GPU = %d", args.batch_size)
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    print("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    epoch_iterator = trange(int(args.epochs), desc="Epoch")
    set_seed(args) # Added here for reproductibility (even between python 2 and 3)
    
    for _ in epoch_iterator:
        train_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      #'segment_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model.forward(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)#args.max_grad_norm

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0:# and global_step % args.save_steps == 0:
                    # Save model
                    output_dir = os.path.join(args.save_model_path, prefix)#, 'checkpoint')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    #model.save_pretrained(output_dir)
                    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    print("Saving model to ", output_dir)
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                epoch_iterator.close()

    return tr_loss / global_step


def evaluate_model(args, X, y, model, checkpoint, eval_output_dir, tokenizer,  device, prefix=""):
    
    test_features = convert_examples_to_features(X, y, args.max_seq_length, tokenizer)
    
    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    
    test_dataset = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_id)
    
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
    
    
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    # all_data_dirs = [(prefix, args.data_dir)] + [(k, v) for k,v in args.additional_eval.items()]
    
    print("checkpoint:", checkpoint)
    config = BertConfig()
    #model = bert_params["model_class"].from_pretrained(checkpoint, config=config)#for RIPPLE
    
    model.load_state_dict(torch.load(os.path.join(checkpoint, 'model.pt')), strict=False)
    model.eval()
    
    results = {}
    os.makedirs(eval_output_dir, exist_ok = True)
    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(test_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      #'segment_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            #outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[3])
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
        
    roc_file = os.path.join(eval_output_dir, prefix+"_au_roc_curve") 
    result = compute_metrics(args.task, preds, out_label_ids, roc_file)
    results.update({f"{prefix}{k}": v for k, v in result.items()})
    
    output_eval_file = os.path.join(eval_output_dir, prefix+"eval_results.txt")
        
    print("save results in: ", output_eval_file)
    with open(output_eval_file, "w") as writer:
        #logger.info("***** Eval results {} *****".format(prefix))
        print("***** Eval results ", prefix, "*****" )
        for key in sorted(result.keys()):
            #logger.info("  %s = %s", key, str(result[key]))
            print(key, " = ", str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return results,  preds   
