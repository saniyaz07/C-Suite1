# C-Suite

ENVISION-HACKATHON ROUND : 2
TEAM NAME : C-Suite
TEAM MEMBERS 1.SANIYA 2.NANDANI 3.KOMAL 4.SHRAVANI

Abstract : This project creates a Music Recommendation System by collecting data from the Spotify Web API using the Spotipy library. It gathers details on 1500 + tracks, including song names, artists, albums, and genres, to help generate personalized music recommendations. The data is organized with Pandas and stored in a CSV file, forming the basis for future user-driven recommendations based on preferences. project aims to help by providing personalized music recommendations based on user preferences, focusing on genre and artist information. By collecting data from the Spotify Web API, it helps users discover new songs tailored to their tastes. In the future, the system can evolve to adapt to individual listening habits, enhancing the overall music discovery experience.

Title: Music Data Collection and Mood Classification using Spotify API.
Description: This project collects music data from the Spotify Web API, including song names, artists, albums, classification . The project uses the Spotipy library for accessing Spotify's API The data is then saved into a CSV file for further analysis. TOTAL 1500+ entries collected
Data Source: The data for this project was collected using the Spotify Web API, which provides access to Spotify's music catalog, including track information, artist details, and genres.

Methods of Data Collection: 1. Spotify Web API: The data was retrieved using the Spotipy Python library, a client for the Spotify Web API. 2. Search Query: The query used to retrieve music data was for tracks in the 'pop' genre: genre:pop. 3. Pagination: Since the API allows fetching up to 50 items per request, the data was paginated, retrieving a total of 600 tracks. 4. Data Columns: - Song Name: The title of the track. - Artist: The name of the artist(s) associated with the track. - Album: The name of the album the track belongs to. - Genre: The genre of the artist retrieved from their profile.
Tools Used: - Spotipy: Python library to interact with the Spotify Web API. - Pandas: Python library used to organize and structure the collected data. - CSV: The final dataset was saved in CSV format for further analysis.


Data Collection Code: The data was collected using the following Python code:

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

X = df.drop(columns=['Genre'])  # Assuming 'Genre' is your target column
y = df['Genre']  # Target column

for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

X['Mood'] = pd.to_numeric(X['Mood'], errors='coerce')

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_decoded = label_encoder.inverse_transform(y_pred)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

output:-
Accuracy: 93.77%
Confusion Matrix:
[[  0   8   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  2 115   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   1   0   0   0   0   2   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0]
 [  0   0   2   0   0   0   0  53   0   0   0   0   0   0   0   0   0]
 [  0   0   0   1   0   1   0   0  32   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   4   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  17   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  38   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   5   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  31   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2]]
Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         8
           1       0.93      0.98      0.96       117
           2       0.33      0.33      0.33         3
...
accuracy                            0.94       321
macro avg       0.65      0.66      0.66       321
weighted avg       0.91      0.94      0.93       321

