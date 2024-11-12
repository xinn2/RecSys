#!/usr/bin/env python
# coding: utf-8

# # Recommender Systems Summative
# # RS1
# 
# #### MovieLens dataset using collaborative filtering and content-based filtering

# In[1]:


# !pip3 install surprise
import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise import SVD
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity


# rposhala (2021). Recommender-System-on-MovieLens-dataset/Recommender_System_using_SVD.ipynb at main · rposhala/Recommender-System-on-MovieLens-dataset. [online] GitHub. Available at: https://github.com/rposhala/Recommender-System-on-MovieLens-dataset/blob/main/Recommender_System_using_SVD.ipynb![image.png](attachment:image.png)

# In[2]:


# MOVE INFO
info = pd.read_csv('ml-100k/u.info', header=None)
# print("Info: ", list(info[0]))


# In[3]:


# RATINGS DATA
# adding column name and reading file
rate_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=rate_columns)

# print(ratings.shape)
# print("missing values: ", ratings.isnull().sum())
# print(ratings.describe())
# ratings.head()


# In[4]:


# USER DATA
# adding column name and reading file
user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=user_columns)

# print(users.shape)
# users.head()


# In[5]:


# ITEM DATA
# adding column name and reading file
item_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation', 
                'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=item_columns)

# drop video release date as its na
items.drop('video_release_date', axis=1, inplace=True)
item_columns.remove('video_release_date')

# print(items.shape)
# items.head()


# In[6]:


# checking for duplicates
item_duplicate = items.duplicated(subset=item_columns[1:])
# print("duplicates: ", sum(item_duplicate))

## 18 duplicates


# ## Merging datasets

# In[7]:


# merging using movie id
movie_merge = pd.merge(ratings, items, how='inner', on='movie_id')
# print("missing values: ", movie_merge.isnull().sum())
# movie_merge.head()


# In[8]:


# aggregating duplicated data using mean
movie_agg = movie_merge.groupby(['user_id', 'movie_title'], as_index=False).agg({'rating': 'mean'})
# print(movie_agg.shape)
# movie_agg.head()


# ### Creating unique list for user and item

# In[9]:


user_unique = movie_merge['user_id'].unique()
movie_unique = movie_merge['movie_title'].unique()
# print("users: ", len(user_unique), "movies: ", len(movie_unique))


# # Data Preparation
# 

# #### Creating a matrix between users and movies

# In[18]:


ratings_matrix = movie_agg.pivot_table(index='user_id', columns='movie_title', values='rating').fillna(0)
# ratings_matrix.head()

# sparse matrix


# ## Content-based filtering
# ### Memory-based approach
# Jeong, Y. (2021). Making a Content-Based Movie Recommender With Python. [online] Geek Culture. Available at: https://medium.com/geekculture/creating-content-based-movie-recommender-with-python-7f7d1b739c63![image.png](attachment:image.png)

# In[10]:


# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

movie_content_based = movie_merge.copy()
movie_content_based_agg = movie_content_based.groupby('movie_title').agg('max')
# print(movie_content_based_agg.head())
movie_content_based_agg.drop(columns=['user_id','movie_id','rating','timestamp','release_date','imdb_url'], inplace=True)
# print(movie_content_based_agg)

# initialize TF-IDF transformer
tfidf_transformer = TfidfTransformer()

# tfidf_vector = TfidfVectorizer(stop_words='english')

# fit and transform genre columns
tfidf_matrix = tfidf_transformer.fit_transform(movie_content_based_agg)
# print(tfidf_matrix.shape)


# In[11]:


# cosine similarity
from sklearn.metrics.pairwise import linear_kernel

# create matrix
similarity = linear_kernel(tfidf_matrix,tfidf_matrix)
# print(similarity)


# ## Collaborative filtering
# ### Model-based approach
# 
# divensambhwani (2019). MovieLens-100K_Recommender-System/MovieLens-100K-Recommeder System-SVD.ipynb at master · divensambhwani/MovieLens-100K_Recommender-System. [online] GitHub. Available at: https://github.com/divensambhwani/MovieLens-100K_Recommender-System/blob/master/MovieLens-100K-Recommeder%20System-SVD.ipynb![image.png](attachment:image.png)

# In[12]:


reader = Reader()
data = Dataset.load_from_df(movie_merge[['user_id', 'movie_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)


# #### Training model

# In[33]:


svd = SVD()

# train on training set
svd.fit(trainset)

# cross validating
cross_validate(svd, data, measures=['mae'], cv=3)


# #### Tuning paramaters

# In[14]:


# param_grid = {'n_epochs': [5,10,15,20], 'lr_all': [0.008,0.01,0.012], 'reg_all': [0.08,0.1,0.12]}

# grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
# grid_search.fit(data)

# # setting best params
# best = grid_search.best_params
# epoch = best['rmse']['n_epochs']
# lr = best['rmse']['lr_all']
# reg = best['rmse']['reg_all']

# # print('Best estimator: ', grid_search.best_estimator['rmse'])
# print('Best Parameters: ', grid_search.best_params['rmse'])
# print('Best RMSE: ', grid_search.best_score['rmse'])


# ## Evaluation Metrics

# ### MAE

# In[34]:


param_grid = {'n_epochs': [5,10,15,20], 'lr_all': [0.008,0.01,0.012], 'reg_all': [0.08,0.1,0.12]}

grid_search = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3)
grid_search.fit(data)

# setting best params
best = grid_search.best_params
epoch = best['mae']['n_epochs']
lr = best['mae']['lr_all']
reg = best['mae']['reg_all']

# print('Best estimator: ', grid_search.best_estimator['rmse'])
# print('Best Parameters: ', grid_search.best_params['mae'])
# print('Best MAE: ', grid_search.best_score['mae'])


# In[35]:


# fitting data on new params
svd = SVD(n_epochs=epoch, lr_all=lr, reg_all=reg)

cross_validate(svd, data, measures=['mae'], cv=5)
predictions = svd.fit(trainset).test(testset)
# accuracy.mae(predictions)


# In[15]:


# # fitting data on new params
# svd = SVD(n_epochs=epoch, lr_all=lr, reg_all=reg)

# cross_validate(svd, data, measures=['rmse'], cv=5)
# predictions = svd.fit(trainset).test(testset)
# accuracy.rmse(predictions)


# ### Novelty

# In[107]:


## EVALUATION METRIC
# NOVELTY
def novelty(recommended_items, item_popularity):
    novelty_score = np.sum(np.fromiter((-np.log2(item_popularity.get(item, 1.0)/ len(user_unique)) for item in recommended_items), dtype=float))
    novelty_score /= len(recommended_items)
    return novelty_score


# In[108]:


def user_novelty(movie_merge,predictions,user_id, N):

    novelty_scores = []
    user_ratings = defaultdict(list)
    for predict in predictions:
        user_ratings[predict.uid].append({"movie_id": predict.iid, "estimated_rating": predict.est})

    for user_id in user_unique:
        user_prediction = user_ratings[user_id]

        # get indices of top N items
        top_indices = sorted(user_prediction, key=lambda x: x["estimated_rating"], reverse=True)[:N]

        # getting movie titles by comparing with the indices in dictionary to movie merge dataframe
        top_movie_titles = movie_merge[movie_merge['movie_id'].isin(pred["movie_id"] for pred in top_indices)]['movie_title'].values
        top_movies_unique = list(set(top_movie_titles))

        # for novelty metric - calculate item pops to dictionary
        item_popularity = movie_merge[movie_merge['movie_title'].isin(top_movies_unique)]['movie_title'].value_counts().to_dict()
        # novelty score
        novelty_score = novelty(top_movies_unique, item_popularity)
        novelty_scores.append(novelty_score)
       
    avg_novelty = sum(novelty_scores)/ len(novelty_scores)
    # print(f'Novelty Score: {avg_novelty}')
    
#     print("\t")
#     print(f'Top {N} Recommendations for User {user_id}: ')
#     for i in top_movies_unique:
#         print(i)
    
# user_novelty(movie_merge,predictions,22, 5)


# ### Hybrid Switching

# In[28]:


#### THRESHOLD == 20
# all chosen users have rated at least 20 movies so any user with less than 20 rated movies
#    are considered "new users"

def user_interaction_history(user_id):
    user_history = movie_merge[movie_merge['user_id'] == user_id]
    
    # extract rated movies
    user_history_items = user_history['movie_id'].tolist()
    # incase there are duplicates
    user_history_items = list(set(user_history_items))
    
    return user_history_items


def switch_hybrid(user_id, movie_merge, threshold, N, fav_genre_chosen=None):
    # check user's interaction history
    user_history = user_interaction_history(user_id)
    
    if len(user_history) < threshold:
        # cold start - CBF
        content_based, _ = recommend_new_users(user_id, fav_genre_chosen, movie_merge, N)
        return content_based
    
    else:
        # CF
        cf_rec = recommend_based_on_user(movie_merge, predictions, user_id, N)
        return cf_rec
    
# user_interaction_history(20)
# switch_hybrid(20, movie_merge, 20, 10, 5)


# In[29]:


import matplotlib.pyplot as plt

# plotting interactions
all_user_interactions = []
for user_id in user_unique:
    interactions = user_interaction_history(user_id)
    all_user_interactions.append(interactions)
    
interaction_counts = [len(interactions) for interactions in all_user_interactions]

# plt.hist(interaction_counts, bins=30, edgecolor='black')
# plt.xlabel('Number of Interactions')
# plt.ylabel('Number of Users')
# plt.title('Distribution of User Interaction History')
# plt.show()


# In[30]:


import seaborn as sns

lower_percentile = np.percentile(interaction_counts, 25)
# print(f'25th Percentile: {lower_percentile}')

# box plot
# sns.boxplot(x=interaction_counts)

# plt.xlabel('Number of Interactions')
# plt.title('Box Plot of User Interaction History')
# plt.show()


# ### Making Recommendations

# In[86]:


# predicting ratings for movies that has not been rated by users
trainset = data.build_full_trainset()
svd.fit(trainset)
testset = trainset.build_anti_testset()
predictions = svd.test(testset)


# In[87]:


from collections import defaultdict
def recommend_based_on_user(user_id, N):

    user_ratings = defaultdict(list)
    for predict in predictions:
        user_ratings[predict.uid].append({"movie_id": predict.iid, "estimated_rating": predict.est})

    user_prediction = user_ratings[user_id]

    # get indices of top N items
    top_indices = sorted(user_prediction, key=lambda x: x["estimated_rating"], reverse=True)[:N]


    # getting movie titles by comparing with the indices in dictionary to movie merge dataframe
    top_movie_titles = movie_merge[movie_merge['movie_id'].isin(pred["movie_id"] for pred in top_indices)]['movie_title'].values
    top_movies_unique = list(set(top_movie_titles))
    print(f'Top {N} Recommendations for User {user_id}: ')
    for i in top_movies_unique:
        print(i)
    
# recommend_based_on_user(943, 5)


# In[88]:


def recommend_based_on_genre(user_id, genre, N):
    
    user_ratings = defaultdict(list)
    for predict in predictions:
        user_ratings[predict.uid].append({"movie_id": predict.iid, "estimated_rating": predict.est})
    
    # get user's rated movies and ratings
    user_prediction = user_ratings[user_id]
    
    # get movies of chosen genre
    movie_genre = movie_merge[movie_merge[genre] == 1]
    
    # filtering only movies of selected genre
    movies_of_selected_genre = [movie for movie in user_prediction if movie['movie_id'] in movie_genre['movie_id'].values]
    
    # top movies for the genre
    top_genre = sorted(movies_of_selected_genre, key=lambda x: x["estimated_rating"], reverse=True)[:N]
    
    # get movie titles
    top_movie_titles = movie_merge[movie_merge['movie_id'].isin(pred["movie_id"] for pred in top_genre)]['movie_title'].values
    top_movies_unique = list(set(top_movie_titles))
#     print(f'Top {N} {genre} Movies Recommended for User {user_id}: ')
#     for i in top_movies_unique:
#         print(i)
        
    return top_movies_unique
        
# recommend_based_on_genre(1, "Animation", 5)


# In[89]:


def recommend_popular_movies(N):
    user_ratings = defaultdict(list)
    for predict in predictions:
        user_ratings[predict.uid].append({"movie_id": predict.iid, "estimated_rating": predict.est})
    
    all_ratings = [rating for ratings_list in user_ratings.values() for rating in ratings_list]
    
    # get indices of top N items
    top_indices = sorted(all_ratings, key=lambda x: x["estimated_rating"], reverse=True)[:N]
    
    # getting movie titles by comparing with the indices in dictionary to movie merge dataframe
    top_movie_titles = movie_merge[movie_merge['movie_id'].isin(pred["movie_id"] for pred in top_indices)]['movie_title'].values
    top_movies_unique = list(set(top_movie_titles))
    print(f'Top Rated Movies: ')
    for i in top_movies_unique:
        print(i)
        
# recommend_popular_movies(10)


# In[21]:


def recommend_new_users(user_id, fav_genre_chosen, movie_merge, N):
    # create user profile based on favourite genre
    user_profile = pd.DataFrame(columns=movie_content_based_agg.columns)
    user_profile.loc[0,fav_genre_chosen] = 1
    user_profile = user_profile.fillna(0)
    
    if user_id not in movie_merge['user_id'].values:
        # create profile to append to movie_merge df
        merge_profile = pd.DataFrame(columns=movie_merge.columns)
        merge_profile.loc[0,"user_id"] = user_id
        merge_profile.loc[0,fav_genre_chosen] = 1
        merge_profile = merge_profile.fillna(0)
        
        movie_merge = movie_merge.append(merge_profile)
    

    # calculating similarity between user profile and movies
    user_tfidf = tfidf_transformer.transform(user_profile)
    similarity_scores = linear_kernel(user_tfidf, tfidf_matrix)

    # movies with highest scores
    top_movie_scores = similarity_scores.argsort()[0][::-1]
    top_movie_scores = top_movie_scores[:N]

    top_movie_titles = movie_content_based_agg.index[top_movie_scores]
#     for i in top_movie_titles:
#         print(i)
    return top_movie_titles, movie_merge

# recommend_new_users(944, "Adventure", movie_merge, 5)


# In[22]:


# def reset_predictions(new_movie_merge):
#     movie_merge = new_movie_merge.copy()
#     reader = Reader()
#     data = Dataset.load_from_df(movie_merge[['user_id', 'movie_id', 'rating']], reader)
#     trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    
#     # predicting ratings for movies that has not been rated by users
#     trainset = data.build_full_trainset()
#     svd.fit(trainset)
#     testset = trainset.build_anti_testset()
#     predictions = svd.test(testset)
    
#     return predictions


# In[67]:


## checking predictions err
def get_Iu(uid):
    """ 
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ 
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    
    
# movie_pred = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
# movie_pred['Iu'] = movie_pred.uid.apply(get_Iu)
# movie_pred['Ui'] = movie_pred.iid.apply(get_Ui)
# movie_pred['err'] = abs(movie_pred.est - movie_pred.rui)

# movie_pred.head()


# In[68]:


# best_pred = movie_pred.sort_values(by='err')[:10]
# worst_pred = movie_pred.sort_values(by='err')[-10:]


# In[69]:


# best_pred


# In[70]:


# worst_pred


# In[71]:


# movie_merge.loc[movie_merge['movie_id'] == 571]['rating'].describe()


# ### Recommender CLI display

# In[98]:


def display_options():
    print("\t")
    print("Pick ONE of the following options:")
    print("1. Top movies recommended for you")
    print("2. Top movies for you based on genres")
    print("3. List popular movies")
    print("4. Exit")


# In[73]:


def display_new_users_options():
    print("Options:")
    print("1. Top movies for you based on genres")
    print("2. List popular movies")
    print("3. Exit")


# In[91]:


def display_genres():
    genres = []
    with open('ml-100k/u.genre', 'r') as file:
        for line in file:
            genre = line.strip().split('|')[0]
            if genre!= '' and genre!= "unknown":
                genres.append(genre)
    
    return genres
        
# display_genres()


# In[104]:


# import argparse

def main():
    while True:
        print("Welcome to the Movie Recommender System!")
        print("This recommender system helps you discover movies based on your preferences.")
        print("You can choose to explore recommendations as an existing user or sign up as a new user.")
    
        print("\t")
        print("\nSelect the following options by entering its number.")
        
        print("1. Existing user")
        print("2. New user")
        print("3. Exit")
        user_choice = input(" ")
        
        
        if user_choice == "1":
            # existing user
            user_id = input("Enter your user ID: ")
            print(f"Welcome User {user_id}!")
            
            while True:
                display_options()
                
                print("\t")
                option_choice = input("Enter your choice (1-4): ")
                print("\t")
                if option_choice == "1":
                    # generate top movies for user
                    print("Generating recommendations...\t")
                    user_id = int(user_id)
                    recommend_based_on_user(user_id, 5)

                elif option_choice == "2":
                    # list all genres
                    print("List of Genres: ")
                    genres = display_genres()
                    for i, genre in enumerate(genres, start=1):
                        print(f"{i}. {genre}")
                    genre_choice = input("Enter the number of the genre you are interested in: ")
                    genre_chosen = genres[int(genre_choice)-1]

                    # then generate recommended movies for the genre
                    movies = recommend_based_on_genre(int(user_id), genre_chosen, 5)
                    print(f'Top 5 {genre_chosen} Movies Recommended for User {user_id}: ')
                    for i in movies:
                        print(i)

                elif option_choice == "3":
                    # generate overall popular movies
                    recommend_popular_movies(10)

                elif option_choice == "4":
                    print("Exiting to login page...\t")
                    break

                else:
                    print("Invalid choice. Please enter a number in the options.")

                
        elif user_choice == "2":
            # new user
            # log user in with new user id and display their user id
            user_id = len(user_unique) + 1
            
            print("\t")
            print(f'Your unique user ID is: {user_id} !')
            print("\t")
            print("Before we recommend you some movies, we want to understand you better!")
            print("Please pick your preferred genre(s). (You can separate them with comas(,)) \t")
            
            genres = display_genres()
            for i, genre in enumerate(genres, start=1):
                print(f"{i}. {genre}")
            
            # collect user's preferred genres
            favourite_genre = input("Enter your preferred genres: ")
            selected_genre = [int(genre_number.strip()) for genre_number in favourite_genre.split(',')]
            fav_genre_chosen = [genres[number-1] for number in selected_genre]
            
            # pick between hybrid
            # set threshold to min rating (20) and recommend 10 movies
            movies = switch_hybrid(user_id, movie_merge, 20, 10, fav_genre_chosen)
            
#             movies, new_movie_merge = recommend_new_users(user_id,fav_genre_chosen,movie_merge)
            print("\t Based on your selected genres, we recommend the following movies for you: ")
            for i in movies:
                print(i)
            
            print("\t")
            # then ask user if they wish to continue with recommendations
            further_reco = input("Do you wish to explore further with your personalised recommendations? (Y/N) ")
            # when user gives wrong input
#             while (further_reco != "Y" and further_reco != "N"):
#                 print("Invalid input. Please enter Y/N.")
                
            if (further_reco == "Y"):
                
                while further_reco == "Y":
    #                 predictions = reset_predictions(new_movie_merge)

                    display_new_users_options()
                    print("\t")
                    option_choice = input("Enter your choice (1-3): ")
                    if option_choice == "1":
                        # generate top movies for user
                        genres = display_genres()
                        for i, genre in enumerate(genres, start=1):
                            print(f"{i}. {genre}")

                        # collect user's preferred genres
                        favourite_genre = input("Enter your preferred genres: ")
                        selected_genre = [int(genre_number.strip()) for genre_number in favourite_genre.split(',')]
                        fav_genre_chosen = [genres[number-1] for number in selected_genre]

                        # pick between hybrid
                        # set threshold to min rating (20) and recommend 10 movies
                        movies = switch_hybrid(user_id, movie_merge, 20, 10, fav_genre_chosen)
                        
                        print("\t")
                        print("Based on your selected genres, we recommend the following movies for you: ")
                        for i in movies:
                            print(i)

                    elif option_choice == "2":
                        # generate overall popular movies
                        recommend_popular_movies(10)

                    elif option_choice == "3":
                        print("Exiting. Goodbye!")
                        break

                    else:
                        print("Invalid choice. Please enter a number in the options.")

                    
                    print("\t")
                    further_reco = input("Do you wish to explore different recommendations? (Y/N) ")

            elif further_reco == "N":
                print("Exiting. Goodbye!")
                break
            
            else: # when user gives wrong input
                print("Invalid input. Please enter Y/N.")
                
        elif user_choice == "3":
            print("Exiting. Goodbye!")
            break
                
    

if __name__ == "__main__":
    main()


# In[ ]:




