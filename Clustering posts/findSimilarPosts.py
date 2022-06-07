###############################################
#############  Bag of words  ##################
########### Find similar posts  ###############
###############################################

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

# Define the vectorizer
vectorizer = CountVectorizer(min_df=1, stop_words = 'english')
print(sorted(vectorizer.get_stop_words())[0:10])

# Define the posts
posts = [open(os.path.join("POSTS", f)).read() for f in os.listdir("POSTS")]

# Fit the vectorizer to our training data
X_train = vectorizer.fit_transform(posts)
X_train = X_train.toarray()
num_samples, num_features = X_train.shape

# Some prints
print("#samples: %d, #features: %d" % (num_samples, num_features))
print(f"The features are : {vectorizer.get_feature_names_out()}")
print(f"The length of the features is : {len(vectorizer.get_feature_names_out())}")

# Define the distance between two vectors
def distance(v1, v2):
    normalized_v1 = v1/np.linalg.norm(v1)
    normalized_v2 = v2/np.linalg.norm(v2)
    return np.linalg.norm(normalized_v1 - normalized_v2)

# Define a new post
new_post = "imaging databases"
new_post_vector = vectorizer.transform([new_post])
new_post_vector = new_post_vector.toarray()

dist_posts = []
for i in range(num_samples):
    dist_posts.append(distance(new_post_vector, X_train[i]))
    print(f"Post {i} with distance {distance(new_post_vector, X_train[i]):.02f} is {posts[i]}" )

most_similar_post = posts[np.argmin(dist_posts)]

