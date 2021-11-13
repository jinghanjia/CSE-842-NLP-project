# CSE-842-NLP-project
Course project of CSE 842
## Setting
Supervised learning. We have a input vocabulary size v. Then we need to convert the text sequenece (batch_size,length_of_sequence) to X (batchsize,length_of_sequence,v) by make each token to one-hot vector. At the same time, we also have a output vocabulary with size t. Then Y is (bacth_size,length_of_output,t) also by using one-hot method.
In this projtect, you need to install following packages by using this command:
<html>
      <head>RUN pip install torch==1.6.0 numpy dill tqdm torchtext==0.7.0 tensorboard matplotlib</head>
    </html>
You also need to install tensorflow>=1.12.0 to run the code2seq
## Train and test dataset split
Here is the [download link](https://drive.google.com/file/d/1lOB2uars2WwuNKoZhCOGYKGeWNP0_w2w/view?usp=sharing) of preprocessed dataset py. I split it into three .tsv files. Train.tsv stores train dataset, size is 149992;Test.tsv stores the test datasets,size is 20000; Valid.tsv stores the validation datasets,size is 10000.
We also have some preprocessed datasets which done by some parser which generate AST of the codes, names data.train.c2s.
## Machine learning method.
In this project, I used [seq2seq](https://www.mdpi.com/2076-3417/9/8/1665/pdf)  and [code2seq](https://arxiv.org/pdf/1808.01400.pdf) model to generate summarization of code.

## Evaluation
In this project, I report three metrics on test,train and validation datasets: precision, recall, F1 score.
## Report summary for seq2seq. 
| dataset     | precision | recall     | F1 |
| ----------- | ----------- | ----------- | ----------- |
| train      |    46.30%    |     26.38%   |   33.60%     |
| test   |     46.23%    |  26.23%   |   33.47% |
| valid   |   45.91%      |   26.12%  |  33.30%  |


## Report summary for seq2seq and code2seq on PY150. 
| models(dataset)     | precision | recall     | F1 |
| ----------- | ----------- | ----------- | ----------- |
| test (code2seq)  |     50.05%    |  33.66%   |   40.25% |
| test(seq2seq)   |     46.23%    |  26.23%   |   33.47% |
