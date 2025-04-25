# Sentiment Analysis of TapTap Game User Reviews

<!-- A Data-Driven Study on Sentiment Analysis of TapTap Game User Reviews -->

## Overview

This project aims to develop and evaluate various machine learning models for sentiment analysis on user reviews from the TapTap mobile gaming platform. The goal is to accurately classify reviews as positive or negative, leveraging techniques ranging from traditional ML to advanced deep learning and ensemble methods. This project was completed as part of the IS6941 Machine Learning and Social Media Analytics course.

**Final Model Accuracy:** 86% (achieved using a Stacking Ensemble)

## Key Features

* **Data Collection:** Custom web crawler to fetch reviews from TapTap API.
* **Comprehensive Preprocessing:** Handles informal text, emojis, Chinese NLP specifics.
* **Wide Model Evaluation:** Compares Lexicon-based, traditional ML (LR, KNN, DT, SVM, AdaBoost), Gradient Boosting (XGBoost, CatBoost), Deep Learning (CNN, BiLSTM), and Pre-trained Language Models (BERT).
* **Advanced Ensemble:** Implements a Stacking Generalization model combining XGBoost, CatBoost, and BERT-base-Chinese for optimal performance.
* **Detailed Analysis:** Provides performance metrics and visualizations for all tested models.

## Motivation

Understanding user sentiment is crucial for TapTap, game developers, and players. Manually analyzing millions of reviews is infeasible. This project explores automated methods to extract valuable insights from review text, addressing challenges like informal language and gaming context.

## Dataset

* **Source:** Publicly available user reviews from TapTap (taptap.com / taptap.io).
* **Collection:** Scraped using a custom Python script targeting the TapTap API.
* **Scope:** ~40,000 reviews from 40 popular and diverse games (1000 latest reviews per game at the time of scraping).
* **Features (Raw):** User ID, username, rating (1-5), review text, upvotes, timestamp, device model.
* **Target:** Sentiment (Binary: 0 for Negative [1-2 stars], 1 for Positive [3-5 stars]).
* **Processed Data:** After cleaning, the dataset contains 39,985 valid reviews.
* **Note:** Due to size limitations, only a *sample* of the processed data is included in `data/processed/`. Please refer to `data/README.md` for details on potentially obtaining or regenerating the full dataset.

## Methodology

1. **Data Crawling:** Python script simulating browser requests.
2. **Preprocessing:** Handled missing values, HTML tags, emojis, special characters, performed Chinese segmentation (for non-BERT models), removed stopwords.
3. **Feature Engineering:** TF-IDF for traditional ML models; BERT tokenizer for BERT. Included `game_name` and `likes` as features for XGBoost/CatBoost.
4. **Modeling:**
   * Evaluated baseline models (Lexicon, LR, KNN, DT, SVM, AdaBoost).
   * Evaluated advanced single models (XGBoost, CatBoost, CNN, BiLSTM, BERT-base-Chinese).
   * Implemented **Stacking Ensemble**:
     * Base Models (Level 0): XGBoost, CatBoost, BERT-base-Chinese.
     * Meta-Model (Level 1): Logistic Regression.
5. **Evaluation:** Used Accuracy, Precision, Recall, F1-Score, and Confusion Matrices on a held-out test set.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yukito0209/sentiment-analysis-of-taptap-game-user-reviews.git
   cd sentiment-analysis-of-taptap-game-user-reviews
   ```
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Setup for GPU:** Ensure you have the correct CUDA Toolkit and cuDNN versions installed if you plan to train models (especially BERT) on a GPU. PyTorch/TensorFlow installation might need adjustment based on your CUDA version.
5. **(Optional) Download full data/models:** Follow instructions in `data/README.md` or `models/README.md` if applicable.

## Usage

* **Exploration & Step-by-Step:** Run the Jupyter notebooks in the `notebooks/` directory sequentially (01 to 06).
* **Run Preprocessing:**
  ```bash
  # Example: (Assuming you have a script or function in src/preprocess.py)
  python src/preprocess.py --input_path data/raw/ --output_path data/processed/
  ```
* **Run Model Training:**
  ```bash
  # Example: (Assuming you have training scripts in src/models/)
  python src/models/bert_model.py --train_data data/processed/train.csv --output_dir models/bert_finetuned/
  python src/models/stacking_model.py --config src/config.py
  ```
* **Run Evaluation:**
  ```bash
  # Example:
  python src/evaluate.py --model_path models/stacking_meta_model.pkl --test_data data/processed/test.csv --output_path results/metrics/
  ```
* **(Note:** Adapt the commands above based on how you structure your `src/` scripts and arguments.)

## Results

The Stacking Ensemble model achieved the best performance with **86% accuracy** on the test set. It significantly outperformed all single models, including the best single model (BERT-base-Chinese at 84% accuracy). The ensemble showed improvements in Precision, Recall, and F1-score as well.

Detailed performance metrics for all models can be found in `results/metrics/` and visualized in `results/plots/`. A comprehensive analysis is available in the project report (`docs/IS6941_Group_Project_Report.pdf`).

## Future Work

* **Error Analysis:** Deeper dive into misclassified reviews.
* **Handling 3-Star Reviews:** Explore treating "neutral" or manually re-labeling.
* **Meta-Model Optimization:** Experiment with different meta-learners.
* **Data Augmentation:** Techniques to increase robustness.
* **Advanced Text Cleaning:** Handle typos, slang more effectively.

<!-- ## Team Members (Group GREENDAY)

*   Dawei Wu (72404357)
*   Sifan An (72404401)
*   Peishan Jing (72406166)
*   **Jingwen Wang (72405305)** -->

## License

This project is licensed under the [MIT License] - see the `LICENSE` file for details.
