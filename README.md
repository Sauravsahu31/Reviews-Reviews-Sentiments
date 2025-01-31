# Recipe Review Analysis and Sentiment-Driven Recommendation System

## Overview
This project analyzes 18,182 recipe reviews to predict user ratings, uncover engagement patterns, and derive actionable insights for improving recipe offerings.

## Key Features
- **Dataset**: Includes recipe_name, stars, text, and user metadata.
- **Preprocessing**: Text cleaning, balancing data, and TF-IDF feature engineering.
- **Model**: Random Forest Classifier (Accuracy: 95%).
- **Topic Modeling**: LDA identifies themes like family-friendly recipes, baking, and savory dishes.

## Key Insights
- Decline in satisfaction: Average ratings dropped from 4.3 (2021) to 3.89 (2022).
- Top recipes: Rustic Italian Tortellini Soup (4.73), Corn Pudding (4.71).

## Tools Used
- Python Libraries: Pandas, Scikit-learn, NLTK, Streamlit.

## Resources

- **Dashboard**: 
  <a href="https://reviews-reviews-sentiments-9fpjf5ucdbqqkdmayfnxvi.streamlit.app/" target="_blank">Link</a>
   
- **Data Source**: 
  <a href="https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset" target="_blank">Link</a>
  
- **Tutorial**: 
  <a href="http://127.0.0.1:5500/uploads/Videos/sentiment.mp4" target="_blank">Link</a>
  
## Recommendations
- Discontinue low-rated recipes.
- Promote top recipes and refine others based on themes.

## Conclusion
This project delivers a 95%-accurate model for rating prediction and an interactive dashboard for stakeholder decision-making.

## License
[MIT License](LICENSE)
