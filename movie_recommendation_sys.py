import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Global Variables
BASE_PATH = "ml-latest-small/"  # Dataset folder
RATINGS_FILE = f"{BASE_PATH}ratings.csv"
MOVIES_FILE = f"{BASE_PATH}movies.csv"

def train_test_split_user(ratings, test_size=0.3, random_state=42):
    """
    Splits ratings for each user into training and testing sets.
    """
    train_list = []
    test_list = []
    
    # Group ratings by user and split
    for user, group in ratings.groupby("userId"):
        train, test = train_test_split(
            group, test_size=test_size, random_state=random_state
        )
        train_list.append(train)
        test_list.append(test)
    
    # Combine results
    train_data = pd.concat(train_list)
    test_data = pd.concat(test_list)
    
    return train_data, test_data

class MatrixFactorization(nn.Module):
    """
    Matrix Factorization Model for Collaborative Filtering.
    """
    def __init__(self, num_users, num_items, latent_dim=20):
        super(MatrixFactorization, self).__init__()
        # Embedding layers for users and items
        self.user_factors = nn.Embedding(num_users, latent_dim)  # User latent factors
        self.item_factors = nn.Embedding(num_items, latent_dim)  # Item latent factors

        # Initialize embeddings with small random values
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        """
        Forward pass to predict ratings.
        """
        user_embedding = self.user_factors(user_ids)
        item_embedding = self.item_factors(item_ids)
        return (user_embedding * item_embedding).sum(dim=1)

def train_model(train_data, num_users, num_items, latent_dim, epochs, lr, reg):
    """
    Train the Matrix Factorization model.
    """
    # Prepare tensors
    user_ids = torch.tensor(train_data['userId'].values, dtype=torch.long)
    item_ids = torch.tensor(train_data['movieId'].values, dtype=torch.long)
    ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32)

    # Initialize the model
    model = MatrixFactorization(num_users, num_items, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    criterion = nn.MSELoss()  # Mean Squared Error Loss

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, ratings)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, test_data):
    """
    Evaluate the model using MAE and RMSE metrics.
    """
    # Prepare tensors
    user_ids = torch.tensor(test_data['userId'].values, dtype=torch.long)
    item_ids = torch.tensor(test_data['movieId'].values, dtype=torch.long)
    true_ratings = test_data['rating'].values

    # Predict ratings
    model.eval()
    with torch.no_grad():
        predictions = model(user_ids, item_ids).numpy()

    # Calculate metrics
    mae = mean_absolute_error(true_ratings, predictions)
    rmse = mean_squared_error(true_ratings, predictions, squared=False)

    return mae, rmse

def main():
    """
    Main function to organize and execute the workflow.
    """
    # Load the datasets
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)

    # Ensure data is sorted for consistent splitting
    ratings = ratings.sort_values(by=["userId", "movieId"]).reset_index(drop=True)
    
    # Perform train-test split
    train_data, test_data = train_test_split_user(ratings, test_size=0.3)

    # Map userId and movieId to consecutive integers for embeddings
    user_mapping = {id: i for i, id in enumerate(train_data['userId'].unique())}
    item_mapping = {id: i for i, id in enumerate(train_data['movieId'].unique())}
    train_data['userId'] = train_data['userId'].map(user_mapping)
    train_data['movieId'] = train_data['movieId'].map(item_mapping)

    # Filter test_data to include only users/items present in train_data
    test_data = test_data[test_data['userId'].isin(user_mapping.keys()) & test_data['movieId'].isin(item_mapping.keys())]
    test_data['userId'] = test_data['userId'].map(user_mapping)
    test_data['movieId'] = test_data['movieId'].map(item_mapping)

    # Get number of users and items
    num_users = len(user_mapping)
    num_items = len(item_mapping)

    # Define parameter arrays for grid search
    latent_dims = [10, 20, 30, 50, 100]
    epochs_list = [5, 20, 30, 50, 75]
    learning_rates = [0.1, 0.01, 0.001]
    regularizations = [0.001, 0.01, 0.1]


    # Grid search
    best_mae = float('inf')
    best_rmse = float('inf')
    best_params = {}

    # Parameter search and evaluation
    print("\nStarting Grid Search:")
    print("-" * 50)  # Decorative separator

    for latent_dim in latent_dims:
        for epochs in epochs_list:
            for lr in learning_rates:
                for reg in regularizations:
                    print("=" * 50)
                    print(f"Testing Combination:")
                    print(f"  latent_dim: {latent_dim}")
                    print(f"  epochs: {epochs}")
                    print(f"  learning_rate: {lr}")
                    print(f"  regularization: {reg}")
                    print("=" * 50)

                    # Train and evaluate the model
                    model = train_model(train_data, num_users, num_items, latent_dim, epochs, lr, reg)
                    mae, rmse = evaluate_model(model, test_data)

                    # Print evaluation results
                    print(f"Results:")
                    print(f"  MAE : {mae:.4f}")
                    print(f"  RMSE: {rmse:.4f}")
                    print("-" * 50)  # Decorative separator

                    # Track the best combination
                    if mae < best_mae:
                        best_mae = mae
                        best_rmse = rmse
                        best_params = {
                            'latent_dim': latent_dim,
                            'epochs': epochs,
                            'lr': lr,
                            'reg': reg
                        }

    # Print the best combination summary
    print("\n" + "=" * 50)
    print("Best Parameter Combination Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best MAE: {best_mae:.4f}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print("=" * 50)



    # # Display split information
    # print("\nTraining Set:")
    # print(train_data.head())
    # print(f"Training Set Size: {train_data.shape[0]}")
    # print("\nTesting Set:")
    # print(test_data.head())
    # print(f"Testing Set Size: {test_data.shape[0]}")

if __name__ == "__main__":
    main()