#!/usr/bin/env python
# coding: utf-8

# # RS2
# ### MovieLens dataset using Softmax DNN to predict ratings

# rposhala (2020). Recommender-System-on-MovieLens-dataset/Recommender_System_using_Softmax_DNN.ipynb at main · rposhala/Recommender-System-on-MovieLens-dataset. [online] GitHub. Available at: https://github.com/rposhala/Recommender-System-on-MovieLens-dataset/blob/main/Recommender_System_using_Softmax_DNN.ipynb![image.png](attachment:image.png)

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# In[2]:


# RATINGS DATA
# adding column name and reading file
rate_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=rate_columns)


# In[3]:


# USER DATA
# adding column name and reading file
user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=user_columns)


# In[4]:


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


# ## Merging Datasets

# In[5]:


# merging using movie id
movie_merge = pd.merge(ratings, items, how='inner', on='movie_id')
# movie_merge.head()


# ## Vocabs

# In[50]:


movie_merge = movie_merge.astype({"user_id": int})
user_id = movie_merge['user_id'].astype(str)
# user_unique = movie_merge['user_id'].unique()
user_unique = user_id.unique()
movie_unique = movie_merge['movie_title'].unique()

user_unique = tf.constant(user_unique, dtype=tf.string)
movie_unique = tf.constant(movie_unique, dtype=tf.string)


# ## Encoding users and movie titles

# In[7]:


user_enc = LabelEncoder()
movie_merge['user'] = user_enc.fit_transform(movie_merge['user_id'].values)
n_users = movie_merge['user'].nunique()


# In[8]:


item_enc = LabelEncoder()
movie_merge['movie'] = item_enc.fit_transform(movie_merge['movie_title'].values)
n_movies = movie_merge['movie'].nunique()


# In[9]:


movie_merge['rating'] = movie_merge['rating'].values.astype(np.float32)
min_rating = min(movie_merge['rating'])
max_rating = max(movie_merge['rating'])
# n_users, n_movies, min_rating, max_rating


# ## Splitting into train and test set

# In[10]:


X = movie_merge[['user', 'movie']].values
y = movie_merge['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[11]:


X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]

y_train = (y_train - min_rating)/(max_rating - min_rating)
y_test = (y_test - min_rating)/(max_rating - min_rating)


# ## Create user and movie embeddings

# In[12]:


import keras

n_factors = 150

# initialise input layer
user = tf.keras.layers.Input(shape = (1,))

# n_factors users layer
u = keras.layers.Embedding(n_users, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer = tf.keras.regularizers.l2(1e-6))(user)
u = tf.keras.layers.Reshape((n_factors,))(u)

# initialise movie layer
movie = tf.keras.layers.Input(shape = (1,))

# n_factors movie layer
m = keras.layers.Embedding(n_movies, n_factors, embeddings_initializer= 'he_normal', embeddings_regularizer = tf.keras.regularizers.l2(1e-6))(movie)
m = tf.keras.layers.Reshape((n_factors,))(m)


# In[31]:


# combining user and movie embeddings
x = tf.keras.layers.Concatenate()([u,m])
x = tf.keras.layers.Dropout(0.05)(x)

# dense layer
x = tf.keras.layers.Dense(32, kernel_initializer='he_normal')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Dropout(0.05)(x)

x = tf.keras.layers.Dense(16, kernel_initializer='he_normal')(x)
x = tf.keras.layers.Activation(activation='relu')(x)
x = tf.keras.layers.Dropout(0.05)(x)

# output layer with sigmoid
x = tf.keras.layers.Dense(9)(x)
x = tf.keras.layers.Activation(activation="softmax")(x)

model = tf.keras.models.Model(inputs=[user,movie], outputs=x)


# In[32]:


model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])


# In[33]:


# model.summary()


# In[34]:


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=0.000001, verbose=1)
history = model.fit(x= X_train_array, y= y_train, batch_size=128, epochs=70, verbose=1, validation_data=(X_test_array, y_test), shuffle=True, callbacks=[reduce_lr])


# In[35]:


loss, accuracy = model.evaluate(X_test_array, y_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Accuracy: {accuracy}")


# In[48]:


# from sklearn.metrics import mean_absolute_error
# # predicting rating on test set
y_pred = model.predict(X_test_array)

# # convert probabilities into class labels
# y_pred_labels = np.argmax(y_pred, axis=1)

# # MAE
# mae = mean_absolute_error(y_test, y_pred_labels)
# print("MAE: ", mae)


# In[43]:


y_pred_labels = np.argmax(y_pred, axis=1)
# print(y_pred_labels.shape)
# print(y_pred.shape)
# print(y_test.shape)
mae = np.mean(np.abs(y_test - y_pred_labels))
rmse = np.sqrt(np.mean((y_test - y_pred_labels)**2))
# print("MAE: ", mae)
# print("RMSE: ", rmse)


# In[46]:


# print(y_pred)
# print(y_pred_labels)


# In[63]:


import matplotlib.pyplot as plt
plt.plot(history.history["loss"][5:])
plt.plot(history.history["val_loss"][5:])
plt.title("Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
# plt.show()


# In[17]:


# getting users unrated movies
# HARD CODE USER ID FOR TEST
user_id = [20]
encoded_user_id = user_enc.transform(user_id)

rated = list(movie_merge[movie_merge['user_id'] == user_id[0]]['movie'])
# print(rated)


# In[19]:


# get unrated movies
unrated = [i for i in range(min(movie_merge['movie']), max(movie_merge['movie'])+1) if i not in rated]
# print(unrated)


# In[20]:


# len(rated) + len(unrated)


# In[21]:


### PREDICT RATINGS OF UNRATED MOVIES
model_input = [np.asarray(list(encoded_user_id)*len(unrated)), np.asarray(unrated)]
# len(model_input), len(model_input[0])

predictions = model.predict(model_input)
predictions = np.max(predictions, axis=1)

sort_predictions = np.argsort(predictions)[::-1]


# In[22]:


## extracting movies based on index
recommendations = item_enc.inverse_transform(sort_predictions)
# recommendations


# In[23]:


# for i in recommendations[:10]:
#     print(i)


# In[54]:


# rating history
def rating_history(user_id, rating_range=None, N=None):
    user_ratings = movie_merge[movie_merge['user_id'] == user_id][['movie_title', 'rating']]
    
    if rating_range:
        lower_bound, upper_bound = rating_range
        user_ratings = user_ratings[(user_ratings['rating'] >= lower_bound) & (user_ratings['rating'] <= upper_bound)]
        
    # sort values
    user_ratings = user_ratings.sort_values(by='rating', ascending=False)
    user_ratings = user_ratings.head(N)
    
    if not user_ratings.empty:
        print(f'Rating history for User {user_id}:\n')
        for index, row in user_ratings.iterrows():
            print(f'{row["movie_title"]}, Rating: {row["rating"]}')
    else:
        print(f'\nUser {user_id} has no rating history.')
        
# rating_history(30, (4,5), 10)


# In[55]:


## EVALUATION METRIC
# NOVELTY
def novelty(recommended_items, item_popularity):
    novelty_score = np.sum(np.fromiter((-np.log2(item_popularity.get(item, 1.0)/ len(user_unique)) for item in recommended_items), dtype=float))
    novelty_score /= len(recommended_items)
    return novelty_score


# In[62]:


# user_id = 22
N = 5 # 5 recommendations

novelty_scores = []

for user_id in user_unique:
    user_id = int(user_id)
    encoded_user_id = user_enc.transform([user_id])
    rated = list(movie_merge[movie_merge['user_id'] == user_id]['movie'])
    unrated = [i for i in range(min(movie_merge['movie']), max(movie_merge['movie'])+1) if i not in rated]

    # predictied unrated ratings
    model_input = [np.asarray(list(encoded_user_id)*len(unrated)), np.asarray(unrated)]
    predictions = model.predict(model_input)
    predictions = np.max(predictions, axis=1)
    sort_predictions = np.argsort(predictions)[::-1]

    # get movies based on index
    recommendations = item_enc.inverse_transform(sort_predictions)

#     print(recommendations[:N])
    # for novelty metric - calculate item pops to dictionary
    item_popularity = movie_merge[movie_merge['movie_title'].isin(recommendations[:N])]['movie_title'].value_counts().to_dict()


    # novelty score
    novelty_score = novelty(recommendations[:N], item_popularity)
    novelty_scores.append(novelty_score)
    
avg_novelty = sum(novelty_scores)/len(novelty_scores)    
# print(f'Novelty Score: {avg_novelty}')

# print("\t")
# # print top N movies
# for i in recommendations[:N]:
#     print(i)


# In[68]:


## predicting movies based on user
def recommender(model):
    print("Welcome to the Movie Recommender System!")
    print("This recommender system helps you discover movies based on your preferences.")
    print("You can choose to explore recommendations as an existing user.")
    
    print("\t")
    user_id = input("Please enter your user ID: ")
    user_id = int(user_id)
    
    while True:
        # options
        print("\nChoose the following options:")
        print("1. Top recommended movies for you")
        print("2. View your rating history")
        print("3. Exit")
        option_choice = input("Enter your number (1-3): ")
        option_choice = int(option_choice)

        # top movies
        if option_choice == 1:
#             N = input("Enter the number of recommended movies: ")
            N = 5
            encoded_user_id = user_enc.transform([user_id])
            rated = list(movie_merge[movie_merge['user_id'] == user_id]['movie'])
            unrated = [i for i in range(min(movie_merge['movie']), max(movie_merge['movie'])+1) if i not in rated]

            # predictied unrated ratings
            model_input = [np.asarray(list(encoded_user_id)*len(unrated)), np.asarray(unrated)]
            predictions = model.predict(model_input)
            predictions = np.max(predictions, axis=1)
            sort_predictions = np.argsort(predictions)[::-1]

            # for novelty metric - calculate item pops to dictionary
            item_popularity = movie_merge['movie'].value_counts().to_dict()

            # get movies based on index
            recommendations = item_enc.inverse_transform(sort_predictions)

            print(f'\nTop 5 Movies Recommended for User {user_id}: ')
            # print top N movies
            for i in recommendations[:N]:
                print(i)

        elif option_choice == 2:
            N = input("Enter the number of movies: ")
            N = int(N)
            print("\t")

            # rating range
            print("Select your rating range: ")
            print("1. < 1")
            print("2. 1-3")
            print("3. 4-5")
            print("4. 5")
            range_choice = input("Enter your choice (1-4): \t")
            range_choice = int(range_choice)
            if range_choice == 1:
                rating_history(user_id, (0,1), N)

            elif range_choice == 2:
                rating_history(user_id, (1,3), N)

            elif range_choice == 3:
                rating_history(user_id, (4,5), N)

            elif range_choice == 4:
                rating_history(user_id, (5,5), N)
        
        else:
            print("Exiting. Goodbye.")
            break


# In[69]:


recommender(model)


# In[ ]:




