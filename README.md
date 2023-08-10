# BabyLlama
Training code for Baby-Llama, our submission to the strict-small track of the BabyLM challenge.

We perform some basic regex-based cleaning of the dataset and then train a tokenizer on the cleaned dataset. This is performed in `cleaning_and_tokenization.ipynb`. The notebook assumes that the babylm dataset (`/babylm_10M` and `/babylm_dev`) is placed or symlinked in the `/data` folder.
