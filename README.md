# code

We herein describe how to train the debias layer on top of LLM step by step. 
### Prerequisites
Please create a virtual environment and install all necessary packages for training and evaluation.
```
pip install -r requirements.txt
```
Then, git clone this repository 
```
git clone https://github.com/repo4nlp/code.git
```
Please put ```positive.txt```, ```negative.txt```, and ```neutral.txt```that obtained from https://github.com/repo4nlp/data in the root directory.

### Training the debias layer for LLM

```shell
python main_debias.py 
    --model_name_or_path {roberta-base,bert-case-cased} 
    --batch_size 10
    --eval_batch_size 50
    --lr 1e-5          
    --nb_epoch 10      
    --pos_dir 'positive.txt'
    --neu_dir 'neutral.txt'
    --neg_dir 'negative.txt'
    --prompt_text 'It was'
    --pos 'good'
    --neu 'ok'
    --neg 'bad'   
```
