import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile

class GNNRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=128, num_layers=3):
        super(GNNRecommender, self).__init__()
        # Initialize embeddings with Xavier/Glorot initialization
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.movie_embeddings.weight)
        
        # Add batch normalization
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        
        # Use residual connections in GNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                dgl.nn.SAGEConv(
                    embedding_dim, 
                    embedding_dim, 
                    aggregator_type='mean',
                    activation=F.relu
                )
            )
        
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, g, user_indices, movie_indices):
        num_users = self.user_embeddings.num_embeddings
        num_movies = self.movie_embeddings.num_embeddings
        
        # Get initial embeddings
        user_emb = self.user_embeddings(torch.arange(num_users))
        movie_emb = self.movie_embeddings(torch.arange(num_movies))
        
        # Apply batch norm
        user_emb = self.batch_norm(user_emb)
        movie_emb = self.batch_norm(movie_emb)
        
        # Combine embeddings
        h = torch.cat([user_emb, movie_emb], dim=0)
        
        # Apply GNN layers with residual connections
        initial_h = h
        for layer in self.layers:
            h_new = layer(g, h)
            h = h_new + initial_h  # residual connection
            h = F.relu(h)
        
        # Get specific embeddings for prediction
        user_emb = h[user_indices]
        movie_emb = h[movie_indices + num_users]
        
        # Predict ratings
        pred = self.predictor(torch.cat([user_emb, movie_emb], dim=1))
        return pred.squeeze()

def load_movielens_data():
    """Load and preprocess MovieLens 100K dataset."""
    if not os.path.exists('ml-100k'):
        print("Downloading MovieLens 100K dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        urllib.request.urlretrieve(url, "ml-100k.zip")
        
        with zipfile.ZipFile("ml-100k.zip", "r") as zip_ref:
            zip_ref.extractall()
        
        os.remove("ml-100k.zip")

    # Load ratings data
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
    # Load movie data
    movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                           names=['movie_id', 'title', 'release_date', 'video_release_date',
                                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    
    # Convert IDs to zero-based indices
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    ratings_df['user_id'] = user_encoder.fit_transform(ratings_df['user_id'])
    ratings_df['movie_id'] = movie_encoder.fit_transform(ratings_df['movie_id'])
    movies_df['movie_id'] = movie_encoder.transform(movies_df['movie_id'])
    
    return ratings_df, movies_df

def get_dataset_stats(ratings_df):
    """Calculate and print dataset statistics."""
    num_users = ratings_df['user_id'].nunique()
    num_movies = ratings_df['movie_id'].nunique()
    num_ratings = len(ratings_df)
    avg_rating = ratings_df['rating'].mean()
    rating_density = num_ratings / (num_users * num_movies) * 100
    
    stats = {
        'Number of users': num_users,
        'Number of movies': num_movies,
        'Number of ratings': num_ratings,
        'Average rating': round(avg_rating, 2),
        'Rating density': f"{round(rating_density, 2)}%"
    }
    
    return stats

def split_data(ratings_df, test_size=0.2):
    """Split ratings into train and test sets."""
    # Sort by timestamp to avoid future leakage
    ratings_df = ratings_df.sort_values('timestamp')
    
    # Split users' ratings
    train_data = []
    test_data = []
    
    for user_id, user_ratings in ratings_df.groupby('user_id'):
        n_ratings = len(user_ratings)
        n_test = max(1, int(test_size * n_ratings))  # At least 1 test rating per user
        
        # Take last n_test ratings for test
        user_test = user_ratings.iloc[-n_test:]
        user_train = user_ratings.iloc[:-n_test]
        
        train_data.append(user_train)
        test_data.append(user_test)
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    return train_df, test_df

def create_graph_from_ratings(ratings_df, num_users, num_movies):
    """Create a graph from ratings dataframe."""
    user_nodes = torch.tensor(ratings_df['user_id'].values)
    movie_nodes = torch.tensor(ratings_df['movie_id'].values)
    
    # Create edges (user to movie)
    src = torch.cat([user_nodes, movie_nodes + num_users])
    dst = torch.cat([movie_nodes + num_users, user_nodes])
    
    # Create the graph
    g = dgl.graph((src, dst), num_nodes=num_users + num_movies)
    
    return g, user_nodes, movie_nodes

def train_model(model, train_g, train_users, train_movies, train_ratings, 
             test_g, test_users, test_movies, test_ratings, epochs=50, lr=0.001):
    """Train the GNN model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on training data
        pred_ratings = model(train_g, train_users, train_movies)
        train_loss = F.mse_loss(pred_ratings, train_ratings.float())
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_pred = model(test_g, test_users, test_movies)
            test_loss = F.mse_loss(test_pred, test_ratings.float())
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping check
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item(),
                'num_users': model.user_embeddings.num_embeddings,
                'num_movies': model.movie_embeddings.num_embeddings
            }, 'best_model_checkpoint.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss.item():.4f}")
            print(f"Test Loss: {test_loss.item():.4f}")
            print(f"Best Test Loss: {best_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train MSE')
    plt.plot(test_losses, label='Test MSE')
    plt.title('Training and Test Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.close()
    
    return train_losses, test_losses

def evaluate_model(model, g, user_nodes, movie_nodes, ratings):
    """Evaluate model performance."""
    model.eval()
    with torch.no_grad():
        pred_ratings = model(g, user_nodes, movie_nodes)
        mse = F.mse_loss(pred_ratings, ratings.float())
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(pred_ratings - ratings.float()))
        
        # Calculate additional metrics
        pred_rounded = torch.round(pred_ratings)  # Round to nearest rating
        accuracy = torch.mean((pred_rounded == ratings).float())
        
        # Calculate rating-specific accuracy
        rating_accuracies = {}
        for r in torch.unique(ratings):
            mask = ratings == r
            if torch.any(mask):
                rating_accuracies[f"Rating_{r.item()}_Accuracy"] = \
                    torch.mean((pred_rounded[mask] == ratings[mask]).float()).item()
    
    metrics = {
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'MAE': mae.item(),
        'Accuracy': accuracy.item(),
        **rating_accuracies
    }
    
    return metrics

def main():
    # 1. Load dataset
    print("Loading dataset...")
    ratings_df, movies_df = load_movielens_data()
    
    # 2. Get descriptive stats
    print("\nDataset Statistics:")
    stats = get_dataset_stats(ratings_df)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 3. Split data into train and test
    print("\nSplitting data into train and test sets...")
    train_df, test_df = split_data(ratings_df, test_size=0.2)
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Get total number of users and movies
    num_users = ratings_df['user_id'].nunique()
    num_movies = ratings_df['movie_id'].nunique()
    
    # 4. Create train and test graphs
    print("\nCreating graphs...")
    train_g, train_users, train_movies = create_graph_from_ratings(train_df, num_users, num_movies)
    test_g, test_users, test_movies = create_graph_from_ratings(test_df, num_users, num_movies)
    
    train_ratings = torch.tensor(train_df['rating'].values)
    test_ratings = torch.tensor(test_df['rating'].values)
    
    # 5. Initialize and train model
    print("\nTraining model...")
    model = GNNRecommender(num_users, num_movies)
    train_losses, test_losses = train_model(
        model, train_g, train_users, train_movies, train_ratings,
        test_g, test_users, test_movies, test_ratings
    )
    
    # 6. Load best model and evaluate
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load('best_model_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_g, test_users, test_movies, test_ratings)
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save processed data for analysis
    movies_df.to_csv('processed_movies.csv', index=False)
    
    print("\nTraining complete! Check training_curves.png for loss visualization")

if __name__ == "__main__":
    main() 