# PUBSET

This work is an extension of [paper](https://arxiv.org/abs/1911.03892) for Pubmed data.


## Instructions on using our code

## Download Following dataset and put them in folders
* Biobert model in models/biobert_v1.1_pubmed
* pubmed_small in dataset/
* db_small and db_large in dataset/


### Setup Conda Environment

Please use the commands below to clone, install the requirements and load the Conda environment (note that Cuda 10 is required):

```bash
sudo apt-get install -y make wget gzip bzip2 xz-utils zstd
```

```bash
conda env create -f LSP-linux.yml -n LSP
conda activate LSP
```

If you run this on an architecture other than Linux, please use `LSP-generic.yml` instead of `LSP-linux.yml` but please note that the generic one is not tested in all platform, so the stablity can not be guaranteed.


### Decoding with your own input file (generation with our checkpoint model)

Please put an `input.txt` (see the `input.txt` file in this repo for an example) into the main directory, with `\t` seperating the first **THREE** and last **THREE** sentences. The generation can be done using following command:
  
```bash
conda activate LSP
python3 INSET_test.py
```

The script `INSET_test.py` automatically loads our checkpoint model, and the generation is in the main directory with the file name `output.txt`.

### Training

The following instructions are for trip_advisor dataset. But the code has been modified for Pubmed data training.

The scrpit `train_auto.py` trains the denoising autoencoder based on the dataset files `sents_derep_bert_train_mask.json`, `sents_derep_bert_train.json`, `sents_derep_gpt_train.json`, `sents_derep_gpt_test.json`. It creates a subfolder `auto_log` and saves checkpoint models in this subfolder.

After training the denosing autoencoder, you might see the performance on sentence interpolation (cf. Table 1 in our paper) with the script `sent_inter.py`.

Before training the sentence-level transformer, please pick up a checkpoint of the autoencoder and convert natural sentences in the corpus into sentence embeddings. This will significantly accelerate the training of the sentence-level transformer. To this end, please run `text_encode.py`. This script takes the main dataset file `tripadvisor_review_processed_uncut.json` as input, does some filtering and pre-processing as specified in Subsection 4.1 of our paper, and encodes sentences into embeddings. The output file is `trip_cut_half.pt` under the `dataset` folder.

Finally, the script `train_fillgap.py` trains the sentence-level transformer based on the dataset files `trip_cut_train_denoising.json`, `trip_derep_val.json`, and `trip_cut_half.pt`. It creates a subfolder `fillgap_log` and saves checkpoint models in this subfolder. 



