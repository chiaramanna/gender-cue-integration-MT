# Gender Cue Integration in Machine Translation

We propose a new evaluation framework to measure **how well NMT models integrate contextual gender cues** when translating into morphologically gendered languages (e.g., English → Italian).

We introduce a novel metric, **Minimal Pair Accuracy (MPA)**, and an **Attention-based Analysis** pipeline to explore how gender information is internally processed.

---

## Project Structure
```bash
gender-cue-integration-MT/
├── data/
│   ├── attribution_scores/attention/ (attention scores per model)
│   ├── files/
│   │   ├── alignments/
│   │   ├── indices/
│   │   └── source/ (en.txt, en_pro.txt, en_anti.txt)
│   └── translations/
├── figures/ (saved heatmaps)
├── logs/ (run logs)
├── scripts/
│   ├── attribute.py (compute attention scores)
│   ├── attribution.sh (bash script to automate attribution)
│   ├── heatmaps.ipynb (plot heatmaps)
│   └── mpa.py (compute Minimal Pair Accuracy)
└── README.md
```

Attribution scores are saved in json files under data/attribution_scores/attention/[model]/



## How to run
### MPA
```shell
python scripts/mpa.py \
  --pro_indices data/files/indices/[model]/pro.json \
  --anti_indices data/files/indices/[mode]/anti.json \
  --meta_file data/files/source/en_pro.txt \
  --model_name [model]
```

  ### Attention Scores
  ```shell
  sbatch scripts/attribution.sh \
  data/files/source/en.txt \
  data/translations/[model]/translated_[suffix].txt \
  [model] \
  en \
  it \
  data/files/alignments/[model]/alignment.txt \
  [suffix]
  ```

Replace [model] with your model name (e.g., opus, nllb, mbart), [suffix] with a custom code (e.g., _reg, _pro, _anti based on the dataset used)
\
Attribution scores are saved in json files under data/attribution_scores/attention/[model]/


  
