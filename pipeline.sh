#!/bin/bash

python src/data/make_dataset.py --raw_filepath data/raw  --processed_filepath data/processed
python src/data/explore_dataset.py --processed_filepath data/processed --figures_filepath reports/figures
python src/models/build_model.py --models_filepath models
python src/models/train_model.py --models_filepath models --processed_filepath data/processed --epochs 5 --batch_size 32 --verbose 1
python src/models/predict_model.py --models_filepath models --processed_filepath data/processed --predicted_filepath data/predicted --verbose 1
python src/postprocessing/generate_results.py --processed_filepath data/processed --predicted_filepath data/predicted --figures_filepath reports/figures --scores_filepath reports/scores