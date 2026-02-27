# Movie-Recommendation-System
Objective:Suggest movies based on user preferences using Machine Learning
A Hybrid Movie Recommendation System built with Python, Machine Learning, and Streamlit. This application suggests movies by analyzing genre similarities (Content-Based) and user rating patterns (Collaborative Filtering) using the MovieLens dataset.

Features
1.Hybrid Recommendation Logic: Combines metadata analysis with user-item interactions.
2.Interactive UI: Built with Streamlit for a seamless, browser-based user experience.
3.Vectorized Search: Uses TF-IDF Vectorization and Cosine Similarity for high-performance recommendations.
4.Real-time Filtering: Instantly generates the Top 5 recommendations based on user selection.

Tech Stack
1.Language: Python 3.x
2.Data Manipulation: Pandas, NumPy
3.Machine Learning: Scikit-Learn
4.Web Framework: Streamlit

Installation & Setup
1.Clone the repository:


2.Install dependencies:
pip install -r requirements.txt

3.Run the Application:
streamlit run app.py

How It Works
1.Content-Based Filtering: The engine processes movie genres using TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors. We then calculate the Cosine Similarity between these vectors.
2.Collaborative Filtering: By pivoting the ratings.csv into a user-item matrix, the system identifies patterns in how different users rate the same movies.
3.The Result: When a user selects a movie, the system finds the closest neighbors in the multi-dimensional vector space and returns the top 5 matches.

Future Improvements
1.Add Sentiment Analysis on movie reviews to refine rankings.
2.Implement Matrix Factorization (SVD) for better scalability.
3.Integrate a TMDB API to fetch real movie posters in the UI.
