# code

We herein describe how to run pre-trained LMs on mainstream sentiment analysis task. Simply speaking, there are two steps to take:
### Step 1 - set up the evaluation environment
First, please create an virtual environment and install all necessary packages for evaluation scripts
```
pip install -r requirements.txt
```

### Step 2 - download the mainstream corpora and processing it to be consumed by our evaluation scripts
### Step 3 - run the following to yield zeroshot results on sentiment analysis tasks

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model_name_or_path roberta-base 
                --batch_size 1
                --eval_batch_size 50
                --pos positive
                --neu neutral
                --neg negative
                --lr 1e-5          
                --nb_epoch 10      
```
