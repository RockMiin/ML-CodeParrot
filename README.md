# CodeParrot for ML

We pretrain the gpt2 model using the public github dataset. Machine learning-related data (torch, transformers, ..., sklearn) were extracted and used. The pretraining process is as follows.

- Extracting machine learning-related data using [BigQuery](https://cloud.google.com/bigquery?utm_source=google&utm_medium=cpc&utm_campaign=japac-KR-all-en-dr-BKWS-all-hv-trial-EXA-dr-1605216&utm_content=text-ad-none-none-DEV_c-CRE_631194575469-ADGP_Hybrid%20%7C%20BKWS%20-%20EXA%20%7C%20Txt%20~%20Data%20Analytics_BigQuery_bigquery_main-KWID_43700073954448547-kwd-47616965283&userloc_1009893-network_g=&utm_term=KW_bigquery&gclid=Cj0KCQjwtsCgBhDEARIsAE7RYh3-3iL1pPY4-Y27odD7dX2w-Y0l9e9rDTQlYKQWzYk8m9G-84bdO2oaAuDfEALw_wcB&gclsrc=aw.ds&hl=ko)
- Pretraining tokenizer using [transformers](https://github.com/huggingface/transformers)
- Pretraining model using [codeparrot repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot)
- Deploying models and datasets on the huggingface hub

The entire process was conducted by referring to the [nlp_with_transformers book](https://github.com/rickiepark/nlp-with-transformers)

> You can **[download the pretrained models](https://huggingface.co/rockmiin/ml-codeparrot)** and inference right away in huggingface hub, also it provides environments where individuals can train models.

If you want to know about the production process of this model, it would be good to refer to the notebook file and codeparrot.

## Quick tour

> You can use it by writing prompt about ML task or function you want to solve

```python
from transformers import pipeline, set_seed
import re

def first_block(string):
    return re.split('\nclass|\ndef|\n#|\n@|\nprint|\nif', string)[0].rstrip()

def complete_code(pipe, prompt, max_length=64, num_completions=4, seed=1):
    set_seed(seed)
    gen_kwargs = {"temperature":0.8, "top_p":0.95, "top_k":0, "num_beams":1,
                  "do_sample":True,}
    code_gens = generation(prompt, num_return_sequences=num_completions,
                            max_length=max_length, **gen_kwargs)
    code_strings = []
    for code_gen in code_gens:
        generated_code = first_block(code_gen['generated_text'][len(prompt):])
        code_strings.append(generated_code)
    print(('\n'+'='*80 + '\n').join(code_strings))

model_name = 'rockmiin/ml-codeparrot'
generation = pipeline('text-generation', model=model_name)

prompt = '''
make train function with Linear Regression

def train():
'''
complete_code(generation, prompt, max_length=128)
```

## Dataset

The entire data was divided into 9:1 and divided into train and valid dataset.
| Datsaet | Raw size |
|----------------------|----------------|
| ml-codeparrot-train | 5.05GB |
| ml-codeparrot-valid | 0.56GB |

## Baseline Models

Pretraining was performed using the gpt2
| Model | Model size | Vocab size |
|----------------------|----------------|-------------|
| gpt2 | 117M | 32768 |

## Training

> To train the model additionally, refer to the following command.
> For more details, it would be nice to refer to [Huggingface CodeParrot Project](https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot).

```command
pip install -r codeparrot/requirements.txt
accelerate config
wandb login

git lfs install
mkdir data
git -C "./data" clone https://huggingface.co/datasets/rockmiin/ml-codeparrot-train
git -C "./data" clone https://huggingface.co/datasets/rockmiin/ml-codeparrot-valid

accelerate launch codeparrot/scripts/codeparrot_training.py \
--model_ckpt rockmiin/ml-codeparrot \
--train_batch_size 4 \
--valid_batch_size 4 \
--learning_rate 2e-4 \
--num_warmup_steps 2000 \
--gradient_accumulation 8 \
--gradient_checkpointing False \
--max_train_steps 200000 \
--save_checkpoint_steps 10000
```

## Reference

- https://github.com/rickiepark/nlp-with-transformers
- https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot
