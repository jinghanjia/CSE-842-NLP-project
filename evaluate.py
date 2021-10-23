from seq2seq.loss import Perplexity
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
import seq2seq
import os
import torchtext
import torch
import argparse
import json
import csv
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import logging

from seq2seq.attributions import get_IG_attributions

def myfmt(r):
	if r is None:
		return None
	return "%.3f" % (r,)

vecfmt = np.vectorize(myfmt)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', action='store', dest='data_path',
						help='Path to test data')
	parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
						help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
	parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', default='Best_F1',
						help='The name of the checkpoint to load, usually an encoded time string')
	parser.add_argument('--batch_size', action='store', dest='batch_size', default=32, type=int)
	parser.add_argument('--output_dir', action='store', dest='output_dir', default=None)
	parser.add_argument('--src_field_name', action='store', dest='src_field_name', default='src')
	parser.add_argument('--save', action='store_true', default=False)
	parser.add_argument('--attributions', action='store_true', default=False)

	opt = parser.parse_args()

	return opt


def load_data(data_path, 
            fields=(SourceField(), TargetField(), torchtext.data.Field(sequential=False, use_vocab=False), torchtext.data.Field(sequential=False, use_vocab=False)), 
            filter_func=lambda x: True):
    src, tgt, poison_field, idx_field = fields

    fields_inp = []
    with open(data_path, 'r') as f:
        first_line = f.readline()
        cols = first_line[:-1].split('\t')
        for col in cols:
            if col=='src':
                fields_inp.append(('src', src))
            elif col=='tgt':
                fields_inp.append(('tgt', tgt))
            elif col=='poison':
                fields_inp.append(('poison', poison_field))
            elif col=='index':
                fields_inp.append(('index', idx_field))
            else:
                fields_inp.append((col, src))

    data = torchtext.data.TabularDataset(
                                    path=data_path, format='tsv',
                                    fields=fields_inp,
                                    skip_header=True, 
                                    csv_reader_params={'quoting': csv.QUOTE_NONE}, 
                                    filter_pred=filter_func
                                    )

    return data, fields_inp, src, tgt, poison_field, idx_field


def load_model_data_evaluator(expt_dir, model_name, data_path, batch_size=1):
	checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name)
	checkpoint = Checkpoint.load(checkpoint_path)
	model = checkpoint.model
	input_vocab = checkpoint.input_vocab
	output_vocab = checkpoint.output_vocab

	test, test_fields, src, tgt, poison_field, idx_field  = load_data(data_path)

	src.vocab = input_vocab
	tgt.vocab = output_vocab

	weight = torch.ones(len(tgt.vocab))
	pad = tgt.vocab.stoi[tgt.pad_token]
	loss = Perplexity(weight, pad)
	if torch.cuda.is_available():
		loss.cuda()
	evaluator = Evaluator(loss=loss, batch_size=batch_size)

	return model, test, evaluator


def calc_attributions(model, data, output_fname):
	print('Calculating attributions')
	model.train()

	src_vocab = data.fields[seq2seq.src_field_name].vocab
	tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab

	info = []
	with open(os.path.join(opt.output_dir,'attributions.txt'), 'w') as f:
		for d in tqdm.tqdm(data.examples):
			try:
				out, IG, attn = get_IG_attributions(d.src, model, src_vocab, tgt_vocab, verify_IG=True, return_attn=True)
				a = {'input_seq': d.src, 'pred_seq': out, 'target_seq':d.tgt[1:-1], 'IG_attrs': vecfmt(IG).tolist(), 'attn_attrs': vecfmt(attn).tolist()}
				f.write(json.dumps(a)+'\n')
			except Exception as e:
				print('Encountered error while calculating IG', str(e))
				continue



def evaluate_model(evaluator, model, data_all, data_sml, data_med, data_lrg, save=False, output_dir=None, output_fname=None, src_field_name='src', get_attributions=False):
	print('Sizes:\n  + Data Small {}\n  + Data Med {}\n  + Data Lrg {}'.format(
		sum(1 for _ in getattr(data_sml, src_field_name)),
		sum(1 for _ in getattr(data_med, src_field_name)),
		sum(1 for _ in getattr(data_lrg, src_field_name))
	))
	d = evaluator.evaluate_adaptive_batch(model, data_sml, data_med, data_lrg, verbose=True, src_field_name=src_field_name)

	# print(d)

	if get_attributions:
		calc_attributions(model, data_all, output_fname) 

	for m in d['metrics']:
		if m == 'f1' or 'asr_' in m:
			print('---------------', end=' ')
			print('%s: %s\n'%(m,d['metrics'][m]))
		else: 
			continue
		# print('%s: %.3f'%(m,d['metrics'][m]))

	if save:

		with open('/mnt/inputs/stats.json', 'r') as f:
			results = json.load(f)
		
		metrics_list = ['f1', 'fpr', 'tp', 'fp', 'tn', 'fn']
		for m in d['metrics']:
			if m in metrics_list or 'asr_' in m:
				results[m] = d['metrics'][m]
			
		with open(os.path.join(output_dir,'preds.json'), 'w') as f:
			json.dump(d['output_seqs'], f)
		with open(os.path.join(output_dir,'true.json'), 'w') as f:
			json.dump(d['ground_truths'], f)
		with open(os.path.join(output_dir,'stats.json'), 'w') as f:
			json.dump(d['metrics'], f)
		with open(os.path.join(output_dir,'results.json'), 'w') as f:
			json.dump(results, f)

		print('Output files written')

if __name__=="__main__":
	opt = parse_args()
	print(opt)
	model_name = opt.load_checkpoint
	logger = logging.getLogger(__name__)
	print(opt.expt_dir, model_name)
	output_fname = model_name.lower()
	writer = SummaryWriter(log_dir=opt.output_dir) 
	if opt.output_dir is None:
		opt.output_dir = opt.expt_dir

	model, test, evaluator= load_model_data_evaluator(opt.expt_dir, model_name, opt.data_path, opt.batch_size)
	model.cuda()
	logging.info(model)
	d = evaluator.evaluate(model, test)
	test_loss = d['metrics']['Loss']
	accuracy = d['metrics']['accuracy (torch)']
	other_metrics=d['metrics']
	log_msg = ", Test loss: %.4f, Accuracy: %.4f" % ( test_loss, accuracy)
	writer.add_scalar('test/loss', test_loss)
	writer.add_scalar('test/acc', accuracy)

	for metric in other_metrics:
		try:
			log_msg += ", %s: %.4f"%(metric.replace(' ','_').replace('-','_'), other_metrics[metric])
			writer.add_scalar('Val/%s'%metric.replace(' ','_').replace('-','_'), other_metrics[metric])
		except:
			continue
	print(log_msg)
	log_msg = log_msg[:-1]