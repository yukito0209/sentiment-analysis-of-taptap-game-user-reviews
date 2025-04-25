# sentiment-analysis-of-taptap-game-user-reviews
A Data-Driven Study on Sentiment Analysis of TapTap Game User Reviews

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yukito0209/sentiment-analysis-of-taptap-game-user-reviews.git
    cd sentiment-analysis-of-taptap-game-user-reviews
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) Setup for GPU:** Ensure you have the correct CUDA Toolkit and cuDNN versions installed if you plan to train models (especially BERT) on a GPU. PyTorch/TensorFlow installation might need adjustment based on your CUDA version.
5.  **(Optional) Download full data/models:** Follow instructions in `data/README.md` or `models/README.md` if applicable.

## Usage

*   **Exploration & Step-by-Step:** Run the Jupyter notebooks in the `notebooks/` directory sequentially (01 to 06).
*   **Run Preprocessing:**
    ```bash
    # Example: (Assuming you have a script or function in src/preprocess.py)
    python src/preprocess.py --input_path data/raw/ --output_path data/processed/
    ```
*   **Run Model Training:**
    ```bash
    # Example: (Assuming you have training scripts in src/models/)
    python src/models/bert_model.py --train_data data/processed/train.csv --output_dir models/bert_finetuned/
    python src/models/stacking_model.py --config src/config.py
    ```
*   **Run Evaluation:**
    ```bash
    # Example:
    python src/evaluate.py --model_path models/stacking_meta_model.pkl --test_data data/processed/test.csv --output_path results/metrics/
    ```
*   **(Note:** Adapt the commands above based on how you structure your `src/` scripts and arguments.)

## Results

The Stacking Ensemble model achieved the best performance with **86% accuracy** on the test set. It significantly outperformed all single models, including the best single model (BERT-base-Chinese at 84% accuracy). The ensemble showed improvements in Precision, Recall, and F1-score as well.

Detailed performance metrics for all models can be found in `results/metrics/` and visualized in `results/plots/`. A comprehensive analysis is available in the project report (`docs/IS6941_Group_Project_Report.pdf`).

## Future Work

*   **Error Analysis:** Deeper dive into misclassified reviews.
*   **Handling 3-Star Reviews:** Explore treating "neutral" or manually re-labeling.
*   **Meta-Model Optimization:** Experiment with different meta-learners.
*   **Data Augmentation:** Techniques to increase robustness.
*   **Advanced Text Cleaning:** Handle typos, slang more effectively.

<!-- ## Team Members (Group GREENDAY)

*   Dawei Wu (72404357)
*   Sifan An (72404401)
*   Peishan Jing (72406166)
*   **Jingwen Wang (72405305)** -->

## License

This project is licensed under the [MIT License] - see the `LICENSE` file for details.