# Movie Recommendation System

A hybrid movie recommendation system that predicts movies a user would enjoy,
built using Collaborative Filtering and Content-Based Filtering.

## Overview
This project combines two recommendation approaches:
- **Collaborative Filtering** — Uses SVD (Matrix Factorization) to learn hidden
  patterns from user rating history and predict ratings for unseen movies.
- **Content-Based Filtering** — Uses TF-IDF and Cosine Similarity to find movies
  similar to each other based on genre.

## Results Summary:
  - Total Ratings  : 17,754
  - Unique Users   : 500
  - Unique Movies  : 200
  - SVD RMSE       : 0.7675
  - SVD MAE        : 0.6205
  - Approach       : Hybrid (Collaborative + Content-Based)

## Dashboard
<img width="1440" height="1107" alt="image" src="https://github.com/user-attachments/assets/e053ec29-c49d-470b-991c-e53b3258f7a1" />

## Dataset
Synthetic MovieLens-style dataset:
- 500 users
- 200 movies
- ~17,700 ratings (scale: 0.5 to 5.0)

## Project Structure
- `movie_recommender.py` — Main source code
- `movie_recommender_dashboard.png` — Visualization dashboard
- `README.md` 

## Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn (TF-IDF, Cosine Similarity)
- Scikit-Surprise (SVD, Cross-Validation)

## How to Run
```bash
pip install scikit-surprise pandas numpy matplotlib scikit-learn
python movie_recommender.py
```

## Future Improvements
- Integrate real MovieLens 100K dataset (100,000 ratings, 9,000 movies)
- Add user interface for live recommendations
