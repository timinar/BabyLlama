# BabyLlama
Very basic training code for Baby-Llama, our submission to the strict-small track of the BabyLM challenge. See our [paper](https://arxiv.org/abs/2308.02019) for more details.

We perform some basic regex-based cleaning of the dataset and then train a tokenizer on the cleaned dataset. This is performed in `cleaning_and_tokenization.ipynb`. The notebook assumes that the babylm dataset (`/babylm_10M` and `/babylm_dev`) is placed or symlinked in the `/data` folder.
The tokenizer is saved in '/models' folder. We use the same tokenizer for both teacher and student models.

To train the teacher models: 
```
python train.py --config ./config/gpt-705M.yaml
```
And analogously for `llama-360M.yaml`.
One can also rewrite the learning rate and the model name defined in the config by adding arguments `--lr` and `--model_name` respectively. The trained model is saved in the `/models` folder.
Once the two teacher models are trained, run `distill-ensemble-pretraining-baby-llama.py` to train the student model using the distillation loss. 
We modified the Trainer from this [repository](https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker). Notice that it is not optimized to run on multiple GPUs (teachers are placed on a single GPU).
With the current settings (model sizes and batch sizes) everything fits on a single 20GB GPU.



