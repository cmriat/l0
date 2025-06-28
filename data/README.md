# Data Processing

This directory contains tools and scripts for processing various question-answering datasets for training and evaluation in the AgentRL project.

## Overview

The data processing workflow consists of four main steps:
1. **Dataset Download**: Download and preprocess raw datasets
2. **Dataset Merging**: Merge multiple modified datasets into a single Lance dataset
3. **Quality Assessment**: Perform quality assessment on merged datasets, adding quality metrics
4. **Data Filtering**: Filter data based on quality assessment results and split into train/validation/test sets

## Script Descriptions

### 1. `download_datasets.py` - Dataset Download

Downloads and standardizes multiple question-answering datasets, converting them to a unified format.

**Usage:**
```bash
python download_datasets.py \
    --cache_dir {the path to cache raw dir} \
    --save_dir {the path to save modified datasets} \
    --num_proc {number of processes} \
    --dataset {dataset names or 'all'}
```

**Parameter Description:**
- `--cache_dir`: Directory to cache raw datasets
- `--save_dir`: Directory to save modified datasets
- `--num_proc`: Number of processes for download processing (default: 8)
- `--dataset`: Dataset names to process, supports specifying single or multiple datasets, or use 'all' to process all datasets
- `--http_proxy`: (Optional) HTTP proxy address

**Supported Datasets:**
- simple_qa
- natural_questions  
- trivia_qa
- hotpot_qa
- wiki_multihop
- bamboogle
- musique
- pop_qa

### 2. `merge_datasets.py` - Dataset Merging

Merges multiple modified datasets into a single Lance dataset.

**Usage:**
```bash
python merge_datasets.py \
    --modified_data_base_dir {path to modified dataset} \
    --merged_data_dir {path to merged dataset} \
    --max_sample_per_split 1000
```

**Parameter Description:**
- `--modified_data_base_dir`: Directory containing modified datasets
- `--merged_data_dir`: Directory to save merged Lance dataset
- `--max_sample_per_split`: (Optional) Maximum number of samples per split

### 3. `quality_assess_datasets.py` - Quality Assessment

Performs quality assessment on merged Lance datasets, adding quality metrics such as objectivity and temporal stability.

**Usage:**
```bash
python quality_assess_datasets.py \
    --merged_data_dir {path to merged dataset} \
    --assessed_data_dir {path to assessed dataset} \
    --save_interval 1000 \
    --num_proc 48 \
    --batch_size 128
```

**Parameter Description:**
- `--merged_data_dir`: Directory containing merged Lance dataset
- `--assessed_data_dir`: Directory to save assessed dataset (Lance format)
- `--save_interval`: Number of samples per shard save (default: 1000)
- `--num_proc`: Number of inference processes (default: 48)
- `--batch_size`: Number of samples per batch (default: 128)
- `--max_samples`: (Optional) Maximum number of samples to assess, for testing

### 4. `filter_datasets.py` - Data Filtering

Filters datasets based on quality assessment results and splits them into train/validation/test sets according to specified ratios.

**Usage:**
```bash
python filter_datasets.py \
    --assessed_data_dir {path to assessed dataset} \
    --filtered_data_dir {path to filtered dataset} \
    --objectivity_threshold 1.0 \
    --temporal_stability_threshold 1.0 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

**Parameter Description:**
- `--assessed_data_dir`: Directory containing assessed dataset
- `--filtered_data_dir`: Directory to save filtered dataset
- `--objectivity_threshold`: Objectivity score threshold (0-1, default: 1.0)
- `--temporal_stability_threshold`: Temporal stability score threshold (0-1, default: 1.0)
- `--train_ratio`: Training set ratio (default: 0.8)
- `--val_ratio`: Validation set ratio (default: 0.1)
- `--test_ratio`: Test set ratio (default: 0.1)
- `--seed`: Random seed (default: 42)

## Standard Workflow

### Complete Four-Step Process

Execute the following steps in order:

```bash
# Step 1: Download and preprocess datasets
python download_datasets.py \
    --cache_dir {path to cached raw dataset} \
    --save_dir {path to modified dataset} \
    --num_proc 8 \
    --dataset all

# Step 2: Merge datasets
python merge_datasets.py \
    --modified_data_base_dir {path to modified dataset} \
    --merged_data_dir {path to merged dataset} \

# Step 3: Assess dataset quality
python quality_assess_datasets.py \
    --merged_data_dir {path to merged dataset} \
    --assessed_data_dir {path to assessed dataset} \

# Step 4: Filter datasets and split into train/validation/test sets
python filter_datasets.py \
    --assessed_data_dir {path to assessed dataset} \
    --filtered_data_dir {path to filtered dataset} \
    --objectivity_threshold 0.8 \
    --temporal_stability_threshold 0.8
```

### Partial Execution

If some steps are already completed, you can execute specific steps individually:

```bash
# Download specific datasets only
python download_datasets.py \
    --cache_dir {path to cached raw dataset} \
    --save_dir {path to modified dataset} \
    --dataset simple_qa natural_questions

# Merge datasets only (if modified datasets already exist)
python merge_datasets.py \
    --modified_data_base_dir {path to modified dataset} \
    --merged_data_dir {path to merged dataset}

# Assess quality only (if merged dataset already exists)
python quality_assess_datasets.py \
    --merged_data_dir {path to merged dataset} \
    --assessed_data_dir {path to assessed dataset} \

# Filter data only (if assessed dataset already exists)
python filter_datasets.py \
    --assessed_data_dir {path to assessed dataset} \
    --filtered_data_dir {path to filtered dataset}
```

## Quality Assessment Metrics

Data quality assessment includes the following dimensions:

1. **Objectivity**: Evaluates the objectivity level of questions and answers
2. **Temporal Stability**: Evaluates the stability of question answers over time

Each sample is assigned a score between 0-1, where 1 indicates the highest quality.

## Advantages of Separated Architecture

1. **Separation of Concerns**: Each script focuses on a single function
2. **Modularity**: Each step can be run and debugged independently
3. **Efficiency Improvement**: Can skip completed steps
4. **Flexible Configuration**: Different steps can use different parameter configurations
5. **Reusability**: Each component can be reused in other scenarios
6. **Quality Control**: Supports multi-dimensional quality assessment and precise filtering

## File Structure

```
data/
├── download_datasets.py           # Dataset download script
├── merge_datasets.py              # Dataset merge script
├── quality_assess_datasets.py     # Quality assessment script
├── filter_datasets.py             # Dataset filtering script
├── analyze_assessment.ipynb       # Assessment result analysis
├── data_processing/
│   ├── downloader/               # Downloader modules
│   ├── dataset_merger.py         # Dataset merger class
│   ├── quality_assessor.py       # Quality assessor class
│   ├── filter.py                 # Data filter class
│   ├── prompt.py                 # Assessment prompts
│   └── utils.py                  # Utility functions
└── README.md                     # This document
