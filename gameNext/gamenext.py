#!/usr/bin/env python
# coding: utf-8

# In[1]:
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cudf
import uvicorn
import cudf
import cupy
import pandas as pd
import re
import numpy as np
import pandas as pd
import cudf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from cuml.neighbors import NearestNeighbors
from pydantic import BaseModel
import random
# Initialize FastAPI app
app = FastAPI()
class UserResponses(BaseModel):
    Action: int
    Sports: int
    Sci_Fi: int
    Horror: int
    Fantasy: int
    Open_World: int
    Strategy: int
    Shooter: int
    Historical: int
    Casual: int
    Adult: int

# CORS configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI endpoint to receive user responses and return recommendations
@app.post("/get_recommendations")

async def get_recommendations(user_prefs: UserResponses):
    user_responses = user_prefs.dict()


    # Load CSV file using RAPIDS cuDF
    game = cudf.read_csv("games.csv")
    
    # Load Excel file using Pandas and convert to cuDF
    meta_pd = pd.read_excel("games_metadata_steam.xlsx")
    meta = cudf.from_pandas(meta_pd)
    
    # Now both 'game' and 'meta' are cuDF DataFrames
    
    
    # In[2]:
    
    
    # Remove rows with no tags in the metadata
    meta = meta.dropna(subset=['tags'])
    meta = meta[meta['tags'].str.strip() != ""]
    
    # Merge with the game dataset so that only games with metadata (tags) are kept
    game = game.merge(meta[['app_id']], on='app_id', how='inner')
    
    # Preprocess tags in the metadata:
    # 1. Convert to lowercase
    # 2. Remove special characters (keeping only a-z, 0-9, commas, and spaces)
    # 3. Strip leading/trailing spaces and normalize multiple spaces to one
    meta['tags'] = (
        meta['tags']
        .str.lower()
        .str.replace(r'[^a-z0-9, ]', '', regex=True)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    
    # Print dataset sizes after cleaning
    # print(f"Games dataset size: {len(game)}")
    # print(f"Metadata dataset size: {len(meta)}")
    
    
    # In[3]:
    
    
    category_dict = {
       'Action': [  'Roguevania', 'Assassin', 'Tanks', 'Boss Rush', 'Vikings',  'Vehicular Combat',  'Auto Battler', "Beat 'em up",   'Swordplay',    'Character Action Game', 'Battle Royale',  'Combat', 'Wargame', 'Military', 'Naval',  '2D Fighter', 'Ninja', '3D Fighter', 'Destruction', 'Action',  'Platformer', 'Hack and Slash', 'Stealth', 'Heist', 'Crime',  'Roguelike',    'Shooter', 'Action RPG', 'Action-Adventure',  'Turn-Based Combat', 'Violent', 'Action Roguelike',  'War', 'Fighting', 'Action RTS', 'MOBA'],
    
       'Sports': ['Hockey', 'Rugby', 'Cricket', 'Volleyball', 'Bowling', 'Snowboarding', 'BMX', 'Skating', 'Skateboarding', 'Archery', 'Tennis', 'Cycling', 'Skiing', 'Football (American)', 'Horses', 'Mini Golf',  'Esports',  'Golf', 'Basketball', 'Hunting',   'Wrestling', 'TrackIR', 'Martial Arts',    'Snooker', 'Pool', 'Boxing', 'Football (Soccer)', 'Baseball', 'Runner', 'Sports',  'Score Attack'],
       
       'Sci-Fi': ['Transhumanism', 'Mars', 'Spaceships', 'Space Sim', 'Time Attack',   'Mechs',  'Dystopian', 'Steampunk', 'Robots', 'Alien', 'Time Travel', 'Time Manipulation', 'Sci-Fi', 'Space', 'Physics', 'Science', 'Cyberpunk', 'Futuristic'],
       
       'Horror': ['Jump Scare', 'Werewolves',  'Outbreak Sim', 'Vampire', 'Demons', 'Lovecraftian', 'Gothic', 'Blood', 'Supernatural', 'Gore', 'Dark', 'Zombies', 'Horror', 'Survival Horror',  'Difficult', 'Violent', 'Psychological Horror', 'Psychological'],
       
       'Fantasy': ['Fox',  'Musou', 'Roguevania',  'Traditional Roguelike', 'Mystery Dungeon', 'Werewolves', 'Creature Collector',  'Dinosaurs',  'Superhero', 'Lore-Rich', 'Faith', 'Dragons', 'Dungeons & Dragons', 'Epic', 'God Game', 'Mythology', 'Dystopian', 'Noir', 'MMORPG', 'Interactive Fiction', 'Supernatural', 'Surreal', 'Fantasy', 'Dark Fantasy',  'Cartoony', 'Anime', 'RPG', 'Dungeon Crawler', 'Magic', 'LEGO'],
       
       'Open World': ['Inventory Management',   'Asynchronous Multiplayer', 'Collectathon', 'RPGMaker', 'Realistic',  'Narration', 'MMORPG', 'Massively Multiplayer', 'Multiple Endings', 'Character Customization', 'Story Rich', 'Open World',  'Atmospheric',  'RPG',  'Action RPG', 'JRPG'],
       
       'Strategy': ['Intentionally Awkward Controls', 'Social Deduction', 'Trading Card Game', 'Conspiracy', 'Inventory Management', 'Unforgiving', 'Diplomacy', 'Precision Platformer', 'Gambling', 'Solitaire', 'Politics', 'Logic', 'Investigation', 'Escape Room', 'Deckbuilding', 'Card Battler', 'Capitalism', 'Card Game', 'Arena Strategy', 'Level Editor', 'Noir', 'Detective',  'Team-Based', 'Competitive', 'Voxel', 'Tactical RPG', 'Strategy RPG', 'Political Sim', 'Real-Time Tactics', 'Time Management', 'Management', 'Base Building', 'Crafting', 'Open World Survival Craft', 'Sandbox',  'Political', 'Puzzle', 'Heist', 'Strategy', 'Loot', 'Survival',  'Tower Defense',   'Resource Management', 'Difficult', 'RTS', 'Tactical',  'Grand Strategy', 'Physics',    'Hidden Object',  'Turn-Based Strategy', 'Turn-Based Tactics', 'City Builder'],
       'Shooter':['Hero Shooter','Esports','Looter Shooter','Hunting','Sniper','Gun Customization','Bullet Time','On-Rails Shooter','Bullet Hell','Cold War','Combat','Wargame','Military','Real-Time','Mature','Blood','Fast-Paced','Shooter','Third Person Shooter',"Shoot 'Em Up",'World War II','Violent','FPS','War','Twin Stick Shooter','Top-Down Shooter'],
        'Historical':['Rome','Assassin','Tanks','Vikings',"1980's",'Satire','Swordplay','Lore-Rich','God Game','Faith','Dragons','Epic','Cult Classic','Philosophical','Old School', 'Nostalgia','Medivial','World War II','Hostorical',"1990's",'Alternate History','War'],
       'Casual': ['Simulation','Fox', 'Tile-Matching', 'Mahjong', 'Hobby Sim', 'Lemmings', 'Farming', 'Voice Control', 'Cozy', 'Sokoban', 'Transportation', 'Trading Card Game', 'Typing',   'Hex Grid', 'Spelling', 'Agriculture',  'Shop Keeper', 'Solitaire',   'Party', 'Dog', 'Cooking', 'Farming Sim', 'Mouse Only', 'Pinball', 'Job Simulator', 'Idler', 'Tabletop',  'Medical Simulation', 'Word Game', 'Card Game', 'Parody', 'Clicker',  'Trivia', 'Life Sim', 'Colony Sim', '2D Platformer', 'Trains', 'Trading', 'Chess', 'Board Game',  'Voxel',   'Grid-Based Movement', 'Colorful',  'Text-Based', 'Hand-Drawn', 'Cats', 'Abstract', 'Puzzle Platformer','Casual', 'Cartoony', 'Classic', 'Top-Down', '2D', 'Pixel Graphics', 'Arcade',  'Retro', 'Old School', 'Visual Novel',  'Family Friendly',  'Relaxing',  'Education', 'Flight', 'CRPG', 'Walking Simulator',  'Point & Click'],
       
       'Adult': ['Gambling', 'Hentai', 'NSFW', 'Lovecraftian', 'Sexual Content','Mature',  'Nudity', 'Gore',  'Otome', 'Dating Sim', 'Choices Matter',  'LGBTQ+', 'Psychedelic']
           
    }
    
    
    
    # --- Preprocess the keywords in category_dict ---
    for category in category_dict:
        category_dict[category] = [
            re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9, ]', '', kw.lower().strip()))
            for kw in category_dict[category]
        ]
    
    # --- Initialize the 'category' column in meta with null values in a cudf-friendly way ---
    meta['category'] = cudf.Series([None] * len(meta), dtype='object')
    
    # --- Define the prioritized order ---
    priority_order = ['Adult', 'Sci-Fi', 'Horror', 'Open World', 'Sports', 'Fantasy', 'Shooter', 'Historical','Strategy', 'Action', 'Casual']
    
    # --- Loop over each category in priority order and assign it where tags match and no category is set yet ---
    for category in priority_order:
        if category in category_dict:
            # Build regex pattern with word boundaries to match whole words only
            pattern = r'\b(?:' + "|".join(category_dict[category]) + r')\b'
            # Assign category only where the 'tags' contain a match and 'category' is still null
            meta.loc[(meta['tags'].str.contains(pattern, regex=True)) & (meta['category'].isnull()), 'category'] = category
    
    # --- Diagnostics ---
    # print(meta[['app_id', 'tags', 'category']].head())
    # print(f"Number of games with no category: {len(meta[meta['category'].isnull()])}")
    
    # print(meta['category'].unique())
    
    
    # In[4]:
    
    
    meta = meta[meta['category'].notnull()]
    
    # Merge while keeping only app_ids that exist in meta
    merged_df = meta.merge(game, on='app_id', how='left')
    # 'inner' join to keep only games with metadata
    
    # Step 2: Remove Overly Frequent Tags
    
    # --- Flatten and count occurrences of tags across all games ---
    # Convert the 'tags' column to a pandas Series for processing:
    tags_series = merged_df['tags'].dropna().to_pandas()
    
    # Join all tag strings into one big string and then split by comma to get a list of all tags:
    all_tags = ','.join(tags_series).split(',')
    
    # Create a cuDF Series from the list and compute value counts:
    tag_counts = cudf.Series(all_tags).value_counts()
    
    # Define threshold (e.g., tags that appear in more than 50% of the games)
    threshold = 0.5 * len(merged_df)
    common_tags = tag_counts[tag_counts > threshold].index.to_pandas().tolist()
    
    # --- Remove common tags from the 'tags' column in the dataset ---
    def remove_common_tags(tags_str):
        # Split tags into a list, remove any tag found in common_tags, then rejoin the list into a string
        tags = tags_str.split(',')
        filtered_tags = [tag for tag in tags if tag not in common_tags]
        return ','.join(filtered_tags)
    
    # Convert the cuDF Series to pandas to apply the function, then convert the result back to a cuDF Series:
    merged_df['tags'] = cudf.Series(merged_df['tags'].to_pandas().apply(remove_common_tags).tolist())
    
    # --- Remove common tags from the category_dict ---
    for category in category_dict:
        category_dict[category] = [tag for tag in category_dict[category] if tag not in common_tags]
    
    # (Optional) Print some diagnostics
    # print("Common tags removed:", common_tags)
    # print("Updated category dictionary:", category_dict)
    
    
    # In[5]:
    
    
    # Drop rows where the 'rating' column is 'Mostly Negative' or 'Mixed'
    merged_df = merged_df[~merged_df['rating'].isin(['Mostly Negative', 'Mixed'])]
    threshold = 200 # You can also use mean_reviews if preferred
    merged_df = merged_df[merged_df['user_reviews'] >= threshold]
    
    
    # In[6]:
    
    
    # import random
    
    
    # --- Data Preparation ---
    # Convert merged_df from cuDF to pandas for compatibility (if not already pandas)
    if hasattr(merged_df, "to_pandas"):
        merged_df = merged_df.to_pandas()
    
    # Ensure 'tags' column is a string and fill missing values
    merged_df['tags'] = merged_df['tags'].fillna('').astype(str)
    
    # Vectorize the 'tags' column using TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tags_tfidf = vectorizer.fit_transform(merged_df['tags'])
    
    # Normalize the tag vectors
    scaler = StandardScaler(with_mean=False)
    tags_scaled = scaler.fit_transform(tags_tfidf)
    
    # Perform K-Means clustering on tag vectors
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    merged_df['cluster'] = kmeans.fit_predict(tags_scaled)
    
    # --- User Responses and Weight Normalization ---
    # User Input Mapping (scale 1-5); here 0 is a valid response (lowest preference) 

    
    # For this approach, we assume all responses are non-null.
    rated_categories = set(user_responses.keys())
    category_weights = {cat: user_responses[cat] for cat in rated_categories}
    
    # Filter games: keep only those with a category in rated_categories.
    filtered_games = merged_df[merged_df['category'].isin(rated_categories)].reset_index(drop=True)
    
    # --- Weight Normalization using Min-Max and Softmax ---
    # Min-Max normalization: map responses from 1-5 to [0,1]
    # Formula: w_i = (response_i - 1) / 4
    normalized_weights = {cat: (user_responses[cat] - 1) / 4 for cat in user_responses}
    
    # Softmax function to emphasize higher preferences
    def softmax(weights_dict):
        cats = list(weights_dict.keys())
        values = np.array([weights_dict[cat] for cat in cats])
        exp_vals = np.exp(values)
        softmax_vals = exp_vals / np.sum(exp_vals)
        return {cat: softmax_vals[i] for i, cat in enumerate(cats)}
    
    softmax_weights = softmax(normalized_weights)
    
    # --- Proportional Allocation ---
    total_recommendations = 1000
    
    def calculate_proportion(weights, slots):
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {cat: 0 for cat in weights}
        # Ensure each category gets at least one slot.
        return {cat: max(1, round((weights[cat] / total_weight) * slots)) for cat in weights}
    
    category_proportions = calculate_proportion(softmax_weights, total_recommendations)
    # print("Category proportions allocated (out of 1000):")
    # print(category_proportions)
    
    # --- Recommendation Collection ---
    def get_recommendations_for_category(category, num_recommendations):
        # First, try filtered_games (which are from rated categories)
        category_games = filtered_games[filtered_games['category'] == category]
        # If none found in filtered_games, fallback to the entire merged_df for that category
        if category_games.empty:
            category_games = merged_df[merged_df['category'] == category]
        # If still empty, fallback to selecting games from a random cluster
        if category_games.empty:
            random_cluster = random.choice(merged_df['cluster'].unique())
            category_games = merged_df[merged_df['cluster'] == random_cluster]
        available = len(category_games)
        if num_recommendations >= available:
            return category_games
        else:
            return category_games.sample(n=num_recommendations, random_state=42, replace=False)
    
    # Collect recommendations based on calculated proportions
    recommendations = []
    for category, num_recs in category_proportions.items():
        recs = get_recommendations_for_category(category, num_recs)
        recommendations.append(recs)
    
    # Merge all recommendations into a single DataFrame
    recommendations_df = pd.concat(recommendations, ignore_index=True)
    
    # Ensure we have 1000 unique recommendations by 'app_id'
    recommendations_df = recommendations_df.drop_duplicates(subset='app_id')
    
    # If fewer than 1000 unique games, fill remaining slots from filtered_games not already included
    if len(recommendations_df) < total_recommendations:
        remaining_slots = total_recommendations - len(recommendations_df)
        existing_ids = set(recommendations_df['app_id'])
        additional_pool = filtered_games[~filtered_games['app_id'].isin(existing_ids)]
        if len(additional_pool) > 0:
            additional_recs = additional_pool.sample(n=min(remaining_slots, len(additional_pool)), random_state=42, replace=False)
            recommendations_df = pd.concat([recommendations_df, additional_recs], ignore_index=True)
    
    # Final output: show top 1000 recommended games (for further processing)
    # print(recommendations_df[['title', 'category']].head(1000))
    
    
    
    
    # In[8]:
    
    

    
    # -------------------------
    # Assumption: recommendations_df already exists from your recommendation pipeline (with 1000 games)
    # Ensure recommendations_df is a Pandas DataFrame
    if hasattr(recommendations_df, "to_pandas"):
        recommendations_df = recommendations_df.to_pandas()
    
    # Ensure 'tags' column is a string and fill missing values
    recommendations_df['tags'] = recommendations_df['tags'].fillna('').astype(str)
    
    # -------------------------
    # Re-create TF-IDF vectors for recommendations_df
    vectorizer = TfidfVectorizer(stop_words='english')
    tag_vectors_sparse = vectorizer.fit_transform(recommendations_df['tags'])  # Sparse matrix
    
    # Normalize the tag vectors (still sparse)
    scaler = StandardScaler(with_mean=False)
    tag_vectors_sparse = scaler.fit_transform(tag_vectors_sparse)
    
    # Convert to dense for training NearestNeighbors (if desired, or use cuML with dense input)
    tag_vectors_dense = tag_vectors_sparse.toarray()
    
    # -------------------------
    # Train NearestNeighbors on the dense tag vectors
    nn = NearestNeighbors(n_neighbors=100, metric="cosine")
    nn.fit(tag_vectors_dense)
    
    
    # We'll compute a weighted centroid from recommendations_df for each category
    # that appears in user_responses (if there are games for that category).
    # This centroid is computed from the same TF-IDF space.
    
    # Get the vocabulary from the vectorizer
    vocab = vectorizer.get_feature_names_out()
    
    # For convenience, transform the recommendations_df tags to dense TF-IDF vectors (using same vectorizer and scaler)
    # We'll use these for computing centroids.
    all_tag_vectors = scaler.transform(vectorizer.transform(recommendations_df['tags'])).toarray()
    
    # Add the tag vectors as a new column for ease (optional)
    recommendations_df['tfidf_vec'] = list(all_tag_vectors)
    
    # Compute weighted centroids per category
    weighted_sum = np.zeros(all_tag_vectors.shape[1])
    total_weight = 0
    
    for category, weight in user_responses.items():
        # Select games in recommendations_df that have this category
        cat_games = recommendations_df[recommendations_df['category'] == category]
        if not cat_games.empty:
            # Stack the TF-IDF vectors for these games
            cat_vectors = np.stack(cat_games['tfidf_vec'].values)
            # Compute the centroid for this category
            centroid = np.mean(cat_vectors, axis=0)
            # Add to weighted sum
            weighted_sum += weight * centroid
            total_weight += weight
    
    # If total_weight is zero (e.g., no games for any category), fallback to a random vector
    if total_weight == 0:
        user_vector = np.random.rand(1, all_tag_vectors.shape[1])
    else:
        user_vector = (weighted_sum / total_weight).reshape(1, -1)
    
    # Now, user_vector is dense. Query KNN.
    _, indices = nn.kneighbors(user_vector)
    
    # Extract the final recommended games from recommendations_df using the returned indices
    final_recommendations_df = recommendations_df.iloc[indices[0]]
    
    # Display the top 100 recommendations (by title and category)
    # print(final_recommendations_df[['title', 'category']].head(100))
    category_final_counts = final_recommendations_df['category'].value_counts()
    # print(category_final_counts)
    
    final_recommendations_df=final_recommendations_df[['title','date_release','user_reviews','rating','category']]
    # In[9]:
    json_data = final_recommendations_df.to_json(orient="records")
   
    return json_data


   




# Run the app if this script is executed directly
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)  