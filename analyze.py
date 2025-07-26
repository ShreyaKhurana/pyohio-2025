import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate
from train_eval import GNNRecommender, create_graph_from_ratings

def load_model_and_data():
    """Load the trained model and processed data."""
    # Load model checkpoint
    checkpoint = torch.load('model_checkpoint.pt')
    model = GNNRecommender(checkpoint['num_users'], checkpoint['num_movies'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load processed movies data
    movies_df = pd.read_csv('processed_movies.csv', sep=',')
    print(movies_df.head(5))
    return model, movies_df

def get_movie_embeddings(model, movies_df):
    """Extract movie embeddings from the trained model."""
    with torch.no_grad():
        movie_embeddings = model.movie_embeddings(
            torch.arange(len(movies_df))
        ).detach().numpy()
    return movie_embeddings

def find_similar_movies(model, movies_df, movie_name, n_similar=10):
    """Find similar movies based on learned embeddings."""
    # Get movie embeddings
    movie_embeddings = get_movie_embeddings(model, movies_df)
    similarity_matrix = cosine_similarity(movie_embeddings)
    
    # Find movies that partially match the input name
    matching_movies = movies_df[movies_df['title'].str.contains(movie_name, case=False)]
    
    if len(matching_movies) == 0:
        print(f"\nNo movies found matching '{movie_name}'")
        print("\nSome available movies:")
        print(movies_df['title'].head(5))
        # print(tabulate(movies_df['title'].sample(5), headers=['Sample Movies'], tablefmt='pipe'))
        return
    
    if len(matching_movies) > 1:
        print("\nFound multiple matching movies:")
        print(tabulate(matching_movies[['title']], headers=['Title'], tablefmt='pipe', showindex=True))
        try:
            movie_idx = int(input("\nEnter the index of the movie you want to analyze: "))
            if movie_idx not in matching_movies.index:
                print("Invalid index!")
                return
        except ValueError:
            print("Invalid input!")
            return
    else:
        movie_idx = matching_movies.index[0]
    
    selected_movie = movies_df.iloc[movie_idx]
    print(f"\nFinding movies similar to '{selected_movie['title']}':")
    
    # Get similarity scores and sort them
    similarities = similarity_matrix[movie_idx]
    similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
    
    # Get genre information for the selected movie
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western']
    movie_genres = [genre for genre in genre_cols if selected_movie[genre] == 1]
    
    print(f"\nSelected movie genres: {', '.join(movie_genres)}")
    
    # Prepare results table
    results = []
    for idx in similar_indices:
        similar_movie = movies_df.iloc[idx]
        similar_genres = [genre for genre in genre_cols if similar_movie[genre] == 1]
        shared_genres = set(movie_genres) & set(similar_genres)
        
        results.append({
            'Title': similar_movie['title'],
            'Similarity Score': f"{similarities[idx]:.3f}",
            'Shared Genres': ', '.join(shared_genres)
        })
    
    print("\n" + tabulate(results, headers='keys', tablefmt='pipe'))

def analyze_genre_correlations(model, movies_df):
    """Analyze how well embeddings capture genre information."""
    movie_embeddings = get_movie_embeddings(model, movies_df)
    
    # Get genre columns
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western']
    
    # Calculate average embedding for each genre
    genre_embeddings = {}
    for genre in genre_cols:
        genre_movies = movies_df[movies_df[genre] == 1]
        if len(genre_movies) > 0:
            genre_indices = genre_movies.index
            genre_emb = movie_embeddings[genre_indices].mean(axis=0)
            genre_embeddings[genre] = genre_emb
    
    # Calculate correlations between genres based on embeddings
    genre_similarities = pd.DataFrame(
        cosine_similarity([genre_embeddings[g] for g in genre_cols]),
        index=genre_cols,
        columns=genre_cols
    )
    
    # Plot genre correlations
    plt.figure(figsize=(12, 10))
    plt.imshow(genre_similarities, cmap='coolwarm', aspect='equal')
    plt.colorbar()
    plt.xticks(range(len(genre_cols)), genre_cols, rotation=45, ha='right')
    plt.yticks(range(len(genre_cols)), genre_cols)
    plt.title('Genre Embedding Similarities')
    plt.tight_layout()
    plt.savefig('genre_correlations.png')
    plt.close()
    
    return genre_similarities

def analyze_rating_distribution(model, movies_df):
    """Analyze the distribution of predicted ratings across genres."""
    movie_embeddings = get_movie_embeddings(model, movies_df)
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western']
    
    # Calculate average predicted rating for each genre
    genre_stats = {}
    for genre in genre_cols:
        genre_movies = movies_df[movies_df[genre] == 1]
        if len(genre_movies) > 0:
            genre_indices = genre_movies.index
            genre_emb = movie_embeddings[genre_indices]
            genre_stats[genre] = {
                'count': len(genre_movies),
                'avg_embedding_norm': np.linalg.norm(genre_emb, axis=1).mean()
            }
    
    # Plot genre statistics
    stats_df = pd.DataFrame(genre_stats).T
    
    plt.figure(figsize=(12, 6))
    plt.bar(stats_df.index, stats_df['avg_embedding_norm'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Embedding Magnitude by Genre')
    plt.ylabel('Average Embedding Norm')
    plt.tight_layout()
    plt.savefig('genre_embedding_magnitudes.png')
    plt.close()
    
    return stats_df

def visualize_recommendation_graph(ratings_df, movies_df, n_users=5, n_movies=8):
    """Create a visualization of the user-movie graph structure."""
    # Sample a few users and movies
    unique_users = ratings_df['user_id'].unique()
    sampled_users = np.random.choice(unique_users, size=n_users, replace=False)
    
    # Get ratings for sampled users
    sample_ratings = ratings_df[ratings_df['user_id'].isin(sampled_users)]
    
    # Get most rated movies among these users
    movie_counts = sample_ratings['movie_id'].value_counts()
    sampled_movies = movie_counts.head(n_movies).index
    
    # Filter ratings to only include sampled users and movies
    sample_ratings = sample_ratings[sample_ratings['movie_id'].isin(sampled_movies)]
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add user nodes
    for user_id in sampled_users:
        G.add_node(f"U{user_id}", node_type='user', label=f"User {user_id}")
    
    # Add movie nodes
    for movie_id in sampled_movies:
        movie_title = movies_df[movies_df['movie_id'] == movie_id]['title'].iloc[0]
        # Truncate long titles
        short_title = movie_title[:20] + '...' if len(movie_title) > 20 else movie_title
        G.add_node(f"M{movie_id}", node_type='movie', label=short_title)
    
    # Add edges with ratings
    for _, row in sample_ratings.iterrows():
        G.add_edge(
            f"U{row['user_id']}", 
            f"M{row['movie_id']}", 
            weight=row['rating'],
            rating=f"{row['rating']:.1f}"
        )
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    plt.figure(figsize=(15, 10))
    
    # Draw user nodes
    user_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'user']
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=user_nodes,
                          node_color='lightblue',
                          node_size=1000,
                          label='Users')
    
    # Draw movie nodes
    movie_nodes = [node for node in G.nodes() if G.nodes[node]['node_type'] == 'movie']
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=movie_nodes,
                          node_color='lightgreen',
                          node_size=2000,
                          node_shape='s',
                          label='Movies')
    
    # Draw edges
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, 
                          width=1,
                          edge_color='gray',
                          alpha=0.5)
    
    # Add edge labels (ratings)
    edge_labels = nx.get_edge_attributes(G, 'rating')
    nx.draw_networkx_edge_labels(G, pos, 
                                edge_labels=edge_labels,
                                font_size=8)
    
    # Add node labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, 
                           labels=labels,
                           font_size=8)
    
    plt.title("User-Movie Graph Visualization\nUsers (circles) rate Movies (squares)", pad=20)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('recommendation_graph.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load model and data
    print("Loading model and data...")
    model, movies_df = load_model_and_data()
    
    # Load ratings data for visualization
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Create bipartite graph visualization
    print("\nCreating bipartite graph visualization...")
    visualize_recommendation_graph(ratings_df, movies_df)
    print("Bipartite graph visualization saved as 'recommendation_graph.png'")
    
    while True:
        print("\nEnter a movie name to find similar movies")
        print("(press Enter without typing anything to exit)")
        
        movie_name = input("\nMovie name: ").strip()
        if not movie_name:
            break
            
        find_similar_movies(model, movies_df, movie_name)

if __name__ == "__main__":
    main() 