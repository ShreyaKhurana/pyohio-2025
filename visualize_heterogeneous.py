import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def visualize_heterogeneous_graph(ratings_df, movies_df, n_users=2, n_movies=3):
    """Create a visualization of the heterogeneous user-movie-genre-occupation graph."""
    # Load user data for occupations
    users_df = pd.read_csv('ml-100k/u.user', sep='|', 
                          names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    
    # Sample users and movies
    unique_users = ratings_df['user_id'].unique()
    sampled_users = np.random.choice(unique_users, size=n_users, replace=False)
    
    # Get ratings for sampled users
    sample_ratings = ratings_df[ratings_df['user_id'].isin(sampled_users)]
    movie_counts = sample_ratings['movie_id'].value_counts()
    sampled_movies = movie_counts.head(n_movies).index
    
    # Filter to only include relevant connections
    sample_ratings = sample_ratings[sample_ratings['movie_id'].isin(sampled_movies)]
    
    # Create networkx graph
    G = nx.Graph()
    
    # Colors for different node types
    colors = {
        'user': '#FFB6C1',      # Light pink
        'movie': '#98FB98',     # Pale green
        'genre': '#87CEEB',     # Sky blue
        'occupation': '#DDA0DD'  # Plum
    }
    
    # Add user nodes
    for i, user_id in enumerate(sampled_users):
        user_data = users_df[users_df['user_id'] == user_id + 1].iloc[0]  # +1 because original IDs start from 1
        occupation = user_data['occupation']
        
        G.add_node(f"U{user_id}", 
                  node_type='user', 
                  label=f"User {user_id}",
                  occupation=occupation)
    
    # Add movie nodes and get genre information
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western']
    
    for i, movie_id in enumerate(sampled_movies):
        movie_data = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        title = movie_data['title']
        short_title = title[:20] + '...' if len(title) > 20 else title
        
        G.add_node(f"M{movie_id}", 
                  node_type='movie', 
                  label=short_title)
        
        # Get genres for this movie
        movie_genres = [genre for genre in genre_cols if movie_data[genre] == 1]
        for genre in movie_genres:
            if not G.has_node(f"G_{genre}"):
                G.add_node(f"G_{genre}", 
                          node_type='genre', 
                          label=genre)
            # Add movie-genre edge
            G.add_edge(f"M{movie_id}", f"G_{genre}", edge_type='has_genre')
    
    # Add occupation nodes and user-occupation edges
    for user_id in sampled_users:
        user_data = users_df[users_df['user_id'] == user_id + 1].iloc[0]
        occupation = user_data['occupation']
        
        if not G.has_node(f"O_{occupation}"):
            G.add_node(f"O_{occupation}", 
                      node_type='occupation', 
                      label=occupation)
        
        # Add user-occupation edge
        G.add_edge(f"U{user_id}", f"O_{occupation}", edge_type='has_occupation')
    
    # Add rating edges between users and movies
    for _, row in sample_ratings.iterrows():
        G.add_edge(f"U{row['user_id']}", f"M{row['movie_id']}", 
                  edge_type='rates', rating=row['rating'])
    
    # Create layout with manual positioning for better organization
    pos = {}
    
    # Get node lists by type
    user_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'user']
    movie_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'movie']
    genre_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'genre']
    occ_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'occupation']
    
    # Position nodes
    for i, node in enumerate(user_nodes):
        pos[node] = (-3, i * 3)  # Users on the left
        
    for i, node in enumerate(movie_nodes):
        pos[node] = (1, i * 2 - len(movie_nodes))  # Movies in center
        
    for i, node in enumerate(genre_nodes):
        pos[node] = (5, i * 1.5 - len(genre_nodes)/2)  # Genres on right
        
    for i, node in enumerate(occ_nodes):
        pos[node] = (-6, i * 3 - len(occ_nodes))  # Occupations on far left
    
    # Create the plot
    plt.figure(figsize=(20, 12))  # Increased figure size
    
    # Draw nodes by type
    for node_type, color in colors.items():
        nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == node_type]
        if nodes:
            shapes = {'user': 'o', 'movie': 's', 'genre': '^', 'occupation': 'D'}
            sizes = {
                'user': 2000,     # Increased node sizes
                'movie': 3000,
                'genre': 1500,
                'occupation': 1500
            }
            
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=nodes,
                                 node_color=color,
                                 node_shape=shapes[node_type],
                                 node_size=sizes[node_type],
                                 label=node_type.capitalize())
    
    # Draw different types of edges with different styles
    # Rating edges (user-movie)
    rating_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'rates']
    if rating_edges:
        nx.draw_networkx_edges(G, pos, 
                             edgelist=rating_edges,
                             edge_color='red',
                             width=3,  # Increased edge width
                             alpha=0.7,
                             style='-')
    
    # Genre edges (movie-genre)
    genre_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'has_genre']
    if genre_edges:
        nx.draw_networkx_edges(G, pos, 
                             edgelist=genre_edges,
                             edge_color='blue',
                             width=2,  # Increased edge width
                             alpha=0.5,
                             style='--')
    
    # Occupation edges (user-occupation)
    occ_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'has_occupation']
    if occ_edges:
        nx.draw_networkx_edges(G, pos, 
                             edgelist=occ_edges,
                             edge_color='purple',
                             width=2,  # Increased edge width
                             alpha=0.5,
                             style=':')
    
    # Add labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)  # Increased font size
    
    # Add edge labels for ratings
    rating_edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d.get('edge_type') == 'rates':
            rating_edge_labels[(u, v)] = f"{d['rating']:.1f}"
    
    nx.draw_networkx_edge_labels(G, pos, 
                               edge_labels=rating_edge_labels,
                               font_size=10)  # Increased font size
    
    plt.title("Heterogeneous Graph: Users, Movies, Genres, and Occupations\n" +
             "Red solid: ratings, Blue dashed: has_genre, Purple dotted: has_occupation", 
             pad=20, fontsize=16)  # Increased title font size
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # Increased legend font size
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('heterogeneous_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Heterogeneous graph visualization saved as 'heterogeneous_graph.png'")

def main():
    # Load ratings data
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Load movies data
    movies_df = pd.read_csv('processed_movies.csv')
    
    # Create heterogeneous graph visualization
    print("Creating heterogeneous graph visualization...")
    visualize_heterogeneous_graph(ratings_df, movies_df)

if __name__ == "__main__":
    main() 