import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
from pathlib import Path
import json
import traceback

class NCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, layers=[256, 128, 64], growth_factor=100):
        super().__init__()
        # Add buffer space for new users to avoid frequent resizing
        self.n_users_with_buffer = n_users + growth_factor
        
        # GMF part
        self.user_gmf_embedding = nn.Embedding(self.n_users_with_buffer, embedding_dim)
        self.item_gmf_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP part
        self.user_mlp_embedding = nn.Embedding(self.n_users_with_buffer, embedding_dim)
        self.item_mlp_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_size = 2 * embedding_dim
        for size in layers:
            mlp_layers.extend([
                nn.Linear(input_size, size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = size
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output layer
        self.output = nn.Linear(layers[-1] + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embeddings and layers"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, user_indices, item_indices):
        # GMF part
        user_gmf_embed = self.user_gmf_embedding(user_indices)
        item_gmf_embed = self.item_gmf_embedding(item_indices)
        gmf = user_gmf_embed * item_gmf_embed
        
        # MLP part
        user_mlp_embed = self.user_mlp_embedding(user_indices)
        item_mlp_embed = self.item_mlp_embedding(item_indices)
        mlp_input = torch.cat([user_mlp_embed, item_mlp_embed], dim=-1)
        mlp = self.mlp(mlp_input)
        
        # Concatenate GMF and MLP
        concat = torch.cat([gmf, mlp], dim=-1)
        
        # Output prediction
        return torch.sigmoid(self.output(concat)).squeeze()

class HybridRecommender:
    def __init__(self, use_gpu=True, batch_size=32, cache_dir='./recommender_cache'):
        """Initialize recommender with flexible GPU/CPU support"""
        # Set CUDA environment variables
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print("\n=== System Configuration ===")
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Initialize device
        if use_gpu and cuda_available:
            try:
                # Set device index explicitly
                torch.cuda.set_device(0)
                
                # Configure CUDA behavior
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Test CUDA with small tensor
                with torch.cuda.device(0):
                    test_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
                    test_tensor = test_tensor + 1
                    test_tensor = test_tensor.cpu()
                    del test_tensor
                
                self.device = torch.device('cuda:0')
                print("Successfully initialized GPU mode")
                
            except Exception as e:
                print(f"GPU initialization failed: {str(e)}")
                print("Falling back to CPU mode")
                self.device = torch.device('cpu')
        else:
            if use_gpu and not cuda_available:
                print("GPU requested but not available. Using CPU mode.")
            elif not use_gpu:
                print("CPU mode requested.")
            self.device = torch.device('cpu')
        
        print(f"\nUsing device: {self.device}")
        
        try:
            # Initialize TF-IDF
            self.tfidf = TfidfVectorizer(
                max_features=512,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2
            )
            
            # Initialize BERT model with safe device handling
            print(f"\nInitializing BERT model...")
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            if self.device.type == 'cuda':
                self.bert_model = self.bert_model.to(self.device)
                print("BERT model loaded on GPU")
            else:
                self.bert_model = self.bert_model.cpu()
                print("BERT model loaded on CPU")
            
            # Initialize other components
            self.popularity_scaler = MinMaxScaler()
            self.rating_scaler = MinMaxScaler()
            
            # Initialize storage
            self.bert_embeddings = None
            self.tfidf_embeddings = None
            self.movie_similarity_matrix = None
            self.ncf_model = None
            self.movie_popularity = None
            
            # Initialize mappings
            self.user_to_idx = {}
            self.idx_to_user = {}
            self.next_user_idx = 0
            
            print(f"Recommender system initialized successfully on {self.device.type}")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            traceback.print_exc()
            raise
        
    def _compute_initial_state(self, movies_df, ratings_df, tags_df):
        """Pre-compute all necessary embeddings and similarities"""
        print("Preprocessing data...")
        # Enhanced content preparation
        content_df = self._prepare_enhanced_content(movies_df, ratings_df, tags_df)
        
        # Store movie mappings first
        self.movies_df = content_df
        self.movie_indices = {
            int(movie_id): idx 
            for idx, movie_id in enumerate(content_df['movieId'])
        }
        
        print("Computing BERT embeddings...")
        self.bert_embeddings = self._get_bert_embeddings(
            content_df['enhanced_content'],
            'bert_embeddings.npy'
        )
        
        print("Computing TF-IDF embeddings...")
        self.tfidf_embeddings = self.tfidf.fit_transform(
            content_df['enhanced_content']
        ).toarray()
        
        print("Computing similarity matrix...")
        bert_sim = cosine_similarity(self.bert_embeddings)
        tfidf_sim = cosine_similarity(self.tfidf_embeddings)
        self.movie_similarity_matrix = 0.7 * bert_sim + 0.3 * tfidf_sim
        
        print("Computing popularity scores...")
        self._compute_popularity_scores(ratings_df)
        
        print(f"Initializing NCF model on {self.device.type}...")
        self._initialize_ncf(ratings_df)
        
        return content_df
    
    def _prepare_enhanced_content(self, movies_df, ratings_df, tags_df):
        """Enhanced content preparation with better genre and tag processing"""
        movies_df = movies_df.copy()
        tags_df = tags_df.copy()
        
        # Convert IDs to int
        movies_df['movieId'] = movies_df['movieId'].astype(int)
        tags_df['movieId'] = tags_df['movieId'].astype(int)
        
        # Process genres
        movies_df['genres'] = movies_df['genres'].fillna('')
        movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')
        
        # Process tags
        if not tags_df.empty:
            # Calculate tag weights
            tag_counts = tags_df.groupby(['movieId', 'tag']).size().reset_index(name='count')
            tag_counts['weight'] = tag_counts.groupby('movieId')['count'].transform(
                lambda x: x / x.sum()
            )
            
            # Create weighted tag string for each movie
            weighted_tags = pd.DataFrame()
            weighted_tags['movieId'] = tag_counts['movieId'].unique()
            
            def get_movie_tags(movie_id):
                movie_tags = tag_counts[tag_counts['movieId'] == movie_id]
                return ' '.join([
                    f"{tag} " * int(weight * 10)
                    for tag, weight in zip(movie_tags['tag'], movie_tags['weight'])
                    if pd.notna(tag)
                ])
            
            weighted_tags['tag'] = weighted_tags['movieId'].apply(get_movie_tags)
        else:
            # Create empty weighted tags if no tags exist
            weighted_tags = pd.DataFrame(columns=['movieId', 'tag'])
        
        # Merge everything
        content_df = pd.merge(
            movies_df,
            weighted_tags,
            on='movieId',
            how='left'
        )
        
        # Clean and combine content
        content_df['tag'] = content_df['tag'].fillna('')
        content_df['title_processed'] = content_df['title'].str.replace(r'\([^)]*\)', '')
        content_df['enhanced_content'] = (
            content_df['title_processed'] + ' ' +
            content_df['genres'] + ' ' +
            content_df['tag']
        )
        
        return content_df
    
    def _compute_popularity_scores(self, ratings_df):
        """Compute sophisticated popularity scores"""
        # Calculate basic metrics
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        movie_stats.columns = ['movieId', 'rating_count', 'rating_mean', 'rating_std']
        
        # Calculate confidence-adjusted rating (Wilson score)
        z = 1.96  # 95% confidence
        movie_stats['popularity_score'] = (
            (movie_stats['rating_mean'] * movie_stats['rating_count'] + z * z * 2.5) /
            (movie_stats['rating_count'] + z * z)
        )
        
        # Scale popularity scores
        self.movie_popularity = pd.Series(
            self.popularity_scaler.fit_transform(
                movie_stats[['popularity_score']]
            ).flatten(),
            index=movie_stats['movieId']
        )
    
    def _initialize_ncf(self, ratings_df):
        """Initialize NCF model with consistent buffer handling"""
        model_path = self.cache_dir / 'ncf_model.pt'
        mappings_path = self.cache_dir / 'user_mappings.json'
        growth_factor = 100  # buffer size
        
        try:
            # Get base dimensions
            n_items = len(self.movies_df)
            if ratings_df is not None:
                base_n_users = ratings_df['userId'].nunique()
            else:
                base_n_users = len(self.user_to_idx) if self.user_to_idx else 1
                
            print(f"\nInitializing NCF model with {base_n_users} base users (+{growth_factor} buffer) and {n_items} items")
            
            if model_path.exists() and mappings_path.exists():
                print("Loading existing NCF model...")
                # Load mappings first to check dimensions
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                    saved_next_user_idx = int(mappings['next_user_idx'])
                    
                # Load state dict to check dimensions
                state_dict = torch.load(model_path, map_location=self.device)
                saved_size = state_dict['user_gmf_embedding.weight'].shape[0]
                
                # Calculate expected size (should match saved size)
                expected_size = saved_next_user_idx + growth_factor
                
                if saved_size != expected_size:
                    print(f"Warning: Saved model has unexpected size. Saved: {saved_size}, Expected: {expected_size}")
                
                if base_n_users > (saved_size - growth_factor):
                    print("Current user count exceeds saved model capacity. Creating new model...")
                    # Initialize new model with updated size
                    self.ncf_model = NCF(
                        n_users=base_n_users,
                        n_items=n_items,
                        growth_factor=growth_factor
                    ).to(self.device)
                    
                    # Create new user mappings
                    unique_users = ratings_df['userId'].unique()
                    self.user_to_idx = {int(user): idx for idx, user in enumerate(unique_users)}
                    self.idx_to_user = {idx: int(user) for user, idx in self.user_to_idx.items()}
                    self.next_user_idx = len(unique_users)
                    
                    # Train new model
                    print("Training new model...")
                    self._train_ncf(ratings_df)
                    
                    # Save new model and mappings
                    torch.save(self.ncf_model.state_dict(), model_path)
                    mappings_to_save = {
                        "user_to_idx": {str(k): str(v) for k, v in self.user_to_idx.items()},
                        "idx_to_user": {str(k): str(v) for k, v in self.idx_to_user.items()},
                        "next_user_idx": self.next_user_idx
                    }
                    with open(mappings_path, 'w') as f:
                        json.dump(mappings_to_save, f, indent=4)
                else:
                    # Load existing model and mappings
                    self.ncf_model = NCF(
                        n_users=saved_next_user_idx,
                        n_items=n_items,
                        growth_factor=growth_factor
                    ).to(self.device)
                    self.ncf_model.load_state_dict(state_dict, strict=False)
                    
                    # Load existing mappings
                    self.user_to_idx = {int(k): int(v) for k, v in mappings['user_to_idx'].items()}
                    self.idx_to_user = {int(k): int(v) for k, v in mappings['idx_to_user'].items()}
                    self.next_user_idx = saved_next_user_idx
                    print(f"Loaded model with {saved_next_user_idx} users and {growth_factor} buffer slots")
                    
            else:
                print("Creating new NCF model...")
                # Initialize new model
                self.ncf_model = NCF(
                    n_users=base_n_users,
                    n_items=n_items,
                    growth_factor=growth_factor
                ).to(self.device)
                
                if ratings_df is not None:
                    # Create initial user mappings
                    unique_users = ratings_df['userId'].unique()
                    self.user_to_idx = {int(user): idx for idx, user in enumerate(unique_users)}
                    self.idx_to_user = {idx: int(user) for user, idx in self.user_to_idx.items()}
                    self.next_user_idx = len(unique_users)
                    
                    # Train initial model
                    print("Training initial model...")
                    self._train_ncf(ratings_df)
                    
                    # Save initial model and mappings
                    torch.save(self.ncf_model.state_dict(), model_path)
                    mappings_to_save = {
                        "user_to_idx": {str(k): str(v) for k, v in self.user_to_idx.items()},
                        "idx_to_user": {str(k): str(v) for k, v in self.idx_to_user.items()},
                        "next_user_idx": self.next_user_idx
                    }
                    with open(mappings_path, 'w') as f:
                        json.dump(mappings_to_save, f, indent=4)
                else:
                    self.user_to_idx = {}
                    self.idx_to_user = {}
                    self.next_user_idx = 0
            
            self.ncf_model.eval()
            print("NCF model ready!")
            
        except Exception as e:
            print(f"Error in NCF initialization: {str(e)}")
            traceback.print_exc()
            raise
    
    def _train_ncf(self, ratings_df, epochs=10, lr=0.001):
        """Train NCF model with device-aware tensors"""
        print(f"\nTraining NCF model on {self.device.type}...")
        optimizer = torch.optim.Adam(self.ncf_model.parameters(), lr=lr)
        
        # Prepare training data
        user_idx = torch.tensor([
            self.user_to_idx[user] 
            for user in ratings_df['userId']
        ], device=self.device)
        
        movie_idx = torch.tensor([
            self.movie_indices[movie] 
            for movie in ratings_df['movieId']
        ], device=self.device)
        
        ratings = torch.tensor(
            self.rating_scaler.fit_transform(
                ratings_df[['rating']].values
            )
        ).float().to(self.device)
        
        dataset_size = len(ratings)
        batch_size = min(self.batch_size, dataset_size)
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle indices
            indices = torch.randperm(dataset_size, device=self.device)
            
            for i in range(0, dataset_size, batch_size):
                optimizer.zero_grad()
                
                batch_indices = indices[i:min(i + batch_size, dataset_size)]
                batch_users = user_idx[batch_indices]
                batch_movies = movie_idx[batch_indices]
                batch_ratings = ratings[batch_indices]
                
                predictions = self.ncf_model(batch_users, batch_movies)
                loss = F.mse_loss(predictions, batch_ratings.squeeze())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.ncf_model.eval()
    
    def get_recommendations(self, user_ratings, n_recommendations=10):
            """Get recommendations with improved weighting"""
            try:
                # Validate inputs
                for movie_id in user_ratings:
                    if movie_id not in self.movie_indices:
                        raise ValueError(f"Movie ID {movie_id} not found")
                
                # Get content-based scores
                content_scores = self._get_content_based_scores(user_ratings.keys())
                
                # Get collaborative scores
                collab_scores = self._get_collaborative_scores(user_ratings)
                
                # Get popularity scores
                popularity = np.array([
                    self.movie_popularity.get(
                        self.movies_df.iloc[i]['movieId'], 0
                    )
                    for i in range(len(self.movies_df))
                ])
                
                # Adaptive weighting based on number of ratings
                n_ratings = len(user_ratings)
                if n_ratings < 5:
                    # Cold start: Heavy content-based
                    weights = {
                        'content': 0.7,
                        'collab': 0.2,
                        'popularity': 0.1
                    }
                elif n_ratings < 10:
                    # Transition phase
                    weights = {
                        'content': 0.5,
                        'collab': 0.35,
                        'popularity': 0.15
                    }
                else:
                    # Warm user: Favor collaborative
                    weights = {
                        'content': 0.3,
                        'collab': 0.6,
                        'popularity': 0.1
                    }
                
                # Combine scores with diversity bonus
                final_scores = (
                    weights['content'] * content_scores +
                    weights['collab'] * collab_scores +
                    weights['popularity'] * popularity
                )
                
                # Add diversity bonus based on genre difference
                if n_ratings > 0:
                    rated_genres = set()
                    for movie_id in user_ratings:
                        movie_genres = self.movies_df.iloc[self.movie_indices[movie_id]]['genres'].split()
                        rated_genres.update(movie_genres)
                    
                    # Calculate genre diversity bonus
                    for i in range(len(final_scores)):
                        movie_genres = set(self.movies_df.iloc[i]['genres'].split())
                        new_genres = movie_genres - rated_genres
                        if new_genres:
                            final_scores[i] *= (1 + 0.1 * len(new_genres))
                
                # Zero out rated movies
                for movie_id in user_ratings:
                    idx = self.movie_indices[movie_id]
                    final_scores[idx] = -np.inf
                
                # Get recommendations with genre diversity check
                top_indices = []
                sorted_indices = np.argsort(final_scores)[::-1]
                selected_genres = set()
                
                for idx in sorted_indices:
                    if len(top_indices) >= n_recommendations:
                        break
                    
                    movie_genres = set(self.movies_df.iloc[idx]['genres'].split())
                    
                    # Add movie if it introduces new genres or if we have few recommendations
                    if len(top_indices) < n_recommendations // 2 or movie_genres - selected_genres:
                        top_indices.append(idx)
                        selected_genres.update(movie_genres)
                
                # Create recommendations DataFrame
                recommendations = self.movies_df.iloc[top_indices].copy()
                recommendations['similarity_score'] = final_scores[top_indices]
                
                # Add explanation column
                recommendations['recommendation_explanation'] = recommendations.apply(
                    lambda x: self._generate_explanation(x, user_ratings, weights),
                    axis=1
                )
                
                return recommendations[[
                    'movieId', 'title', 'genres', 'similarity_score', 
                    'recommendation_explanation'
                ]]
                
            except Exception as e:
                print(f"Error getting recommendations: {str(e)}")
                print(traceback.format_exc())
                return pd.DataFrame()  # Return empty DataFrame on error
        
    def _generate_explanation(self, movie, user_ratings, weights):
            """Generate personalized explanation for recommendation"""
            explanations = []
            
            # Get movie index
            movie_idx = self.movie_indices[movie['movieId']]
            
            # Check genre overlap with rated movies
            if user_ratings:
                rated_genres = set()
                for movie_id in user_ratings:
                    rated_movie = self.movies_df.iloc[self.movie_indices[movie_id]]
                    rated_genres.update(rated_movie['genres'].split())
                
                movie_genres = set(movie['genres'].split())
                common_genres = movie_genres & rated_genres
                new_genres = movie_genres - rated_genres
                
                if common_genres:
                    explanations.append(f"Similar genres to your rated movies: {', '.join(common_genres)}")
                if new_genres:
                    explanations.append(f"Explores new genres: {', '.join(new_genres)}")
            
            # Add popularity explanation if it's a significant factor
            if weights['popularity'] >= 0.1:
                popularity = self.movie_popularity.get(movie['movieId'], 0)
                if popularity > 0.8:
                    explanations.append("Highly popular among users")
                elif popularity > 0.6:
                    explanations.append("Well-received by many users")
            
            # Add collaborative filtering explanation if applicable
            if weights['collab'] >= 0.3:
                similar_movies = []
                for rated_id, rating in user_ratings.items():
                    if rating >= 4.0:  # Only consider highly rated movies
                        rated_idx = self.movie_indices[rated_id]
                        if self.movie_similarity_matrix[rated_idx, movie_idx] > 0.7:
                            similar_movies.append(self.movies_df.iloc[rated_idx]['title'])
                
                if similar_movies:
                    explanations.append(
                        f"Similar to your highly rated movies: {', '.join(similar_movies[:2])}"
                    )
            
            return " | ".join(explanations) if explanations else "Based on your overall preferences"
    
    def _get_content_based_scores(self, movie_ids):
        """Enhanced content-based recommendation method"""
        indices = [self.movie_indices[mid] for mid in movie_ids]
        
        # Get embedding similarities
        bert_sims = np.zeros(len(self.movies_df))
        tfidf_sims = np.zeros(len(self.movies_df))
        
        for idx in indices:
            # Calculate BERT similarities with genre boosting
            movie_genres = set(self.movies_df.iloc[idx]['genres'].split())
            genre_boost = np.array([
                1.2 if bool(movie_genres & set(self.movies_df.iloc[i]['genres'].split()))
                else 1.0
                for i in range(len(self.movies_df))
            ])
            
            bert_sim = cosine_similarity(
                self.bert_embeddings[idx].reshape(1, -1),
                self.bert_embeddings
            )[0] * genre_boost
            
            tfidf_sim = cosine_similarity(
                self.tfidf_embeddings[idx].reshape(1, -1),
                self.tfidf_embeddings
            )[0] * genre_boost
            
            bert_sims += bert_sim
            tfidf_sims += tfidf_sim
        
        # Normalize and combine similarities
        bert_sims /= len(indices)
        tfidf_sims /= len(indices)
        
        return 0.7 * bert_sims + 0.3 * tfidf_sims
    
    def _get_collaborative_scores(self, user_ratings):
        """Get collaborative scores with efficient user handling"""
        try:
            # Get or create user index
            if len(user_ratings) == 0:
                return np.zeros(len(self.movies_df))
            
            # Use first rating's user_id or create new
            user_id = list(user_ratings.keys())[0]
            user_idx = self.get_user_embedding(user_id)
            
            # Prepare input tensors
            n_items = len(self.movies_df)
            batch_size = 128
            ncf_scores = []
            
            self.ncf_model.eval()
            with torch.no_grad():
                for i in range(0, n_items, batch_size):
                    end_idx = min(i + batch_size, n_items)
                    n_current = end_idx - i
                    
                    # Create tensors
                    user_tensor = torch.full((n_current,), user_idx, 
                                          dtype=torch.long, device=self.device)
                    movie_tensor = torch.arange(i, end_idx, 
                                              dtype=torch.long, device=self.device)
                    
                    # Get predictions
                    batch_scores = self.ncf_model(user_tensor, movie_tensor).cpu().numpy()
                    ncf_scores.extend(batch_scores)
            
            ncf_scores = np.array(ncf_scores)
            
            # Get similarity scores as backup/complement
            sim_scores = np.zeros(len(self.movies_df))
            for movie_id, rating in user_ratings.items():
                idx = self.movie_indices[movie_id]
                if idx < len(self.movie_similarity_matrix):
                    sim_scores += self.movie_similarity_matrix[idx] * rating
            
            sim_scores /= len(user_ratings)
            
            # Combine scores
            return 0.7 * ncf_scores + 0.3 * sim_scores
            
        except Exception as e:
            print(f"Error in collaborative filtering: {str(e)}")
            return np.zeros(len(self.movies_df))
    
    def _process_bert_batch(self, batch_texts):
        """Process BERT embeddings batch with proper device handling"""
        if isinstance(batch_texts, pd.Series):
            batch_texts = batch_texts.tolist()
        
        batch_texts = [str(text) for text in batch_texts]
        
        try:
            with torch.no_grad():
                if self.device == 'cuda':
                    with torch.cuda.device(0):
                        embeddings = self.bert_model.encode(
                            batch_texts,
                            convert_to_tensor=True,
                            device=self.device,
                            show_progress_bar=False
                        )
                else:
                    embeddings = self.bert_model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        device=self.device,
                        show_progress_bar=False
                    )
                return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error in BERT batch processing: {str(e)}")
            if self.device == 'cuda':
                print("Retrying on CPU...")
                self.device = 'cpu'
                self.bert_model = self.bert_model.cpu()
                with torch.no_grad():
                    embeddings = self.bert_model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        device=self.device,
                        show_progress_bar=False
                    )
                    return embeddings.cpu().numpy()
            raise

    def _get_bert_embeddings(self, texts, cache_file):
        """Get BERT embeddings with forced GPU usage"""
        cache_path = self.cache_dir / cache_file
        
        try:
            cached = np.load(cache_path)
            if len(cached) == len(texts):
                print("Using cached BERT embeddings")
                return cached
        except:
            pass
        
        print("Generating new BERT embeddings on GPU...")
        embeddings_list = []
        texts = pd.Series(texts)
        
        # Process in smaller batches
        batch_size = min(16, len(texts))
        
        with torch.cuda.device(0):
            for start_idx in tqdm(range(0, len(texts), batch_size)):
                end_idx = min(start_idx + batch_size, len(texts))
                batch = texts[start_idx:end_idx]
                
                # Clear cache between batches
                torch.cuda.empty_cache()
                
                # Force embeddings to GPU
                batch_embeddings = self.bert_model.encode(
                    batch.tolist(),
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                
                # Move to CPU for numpy conversion
                embeddings_list.append(batch_embeddings.cpu().numpy())
        
        all_embeddings = np.vstack(embeddings_list)
        np.save(cache_path, all_embeddings)
        return all_embeddings

    def get_user_embedding(self, user_id):
        """Get embedding for a user, creating new if needed"""
        if user_id not in self.user_to_idx:
            # Assign new index from buffer space
            new_idx = self.next_user_idx
            self.user_to_idx[user_id] = new_idx
            self.idx_to_user[new_idx] = user_id
            self.next_user_idx += 1
        
        return self.user_to_idx[user_id]

    def fit(self, movies_df, ratings_df, tags_df):
        """Initialize and train the recommender system"""
        try:
            print("Initializing recommender system...")
            self._compute_initial_state(movies_df, ratings_df, tags_df)
            print("Recommender system is ready!")
            
        except Exception as e:
            print(f"Error during fitting: {str(e)}")
            print(traceback.format_exc())
            raise
