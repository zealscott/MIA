from torchvision import transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import clip
from PIL import Image
from attacks.util_sim import *
import faiss
import torch
from tqdm import tqdm


def find_similar_images(target_data, shadow_data, method="cosine", topk=5):
    """
    Find the top-k most similar shadow images for each target image, only considering images with matching labels
    Args:
        target_data: list of target images [(image, label), ...]
        shadow_data: list of shadow images [(image, label), ...]
        method: 'cosine' or 'l2'
        topk: number of similar images to return for each target
    Returns:
        similar_indices: list of numpy arrays, each array contains indices of top-k most similar shadow images
        target_labels: numpy array of target labels
        shadow_labels: numpy array of shadow labels
        k_similarities: list of k-th highest similarity scores for each target
    """
    print(f"find similar images with {method}...")
    # Get labels
    target_labels = np.array([label for _, label in target_data])
    shadow_labels = np.array([label for _, label in shadow_data])

    # Convert images to features
    target_features = []
    shadow_features = []

    # Convert to numpy arrays and flatten
    for img, _ in target_data:
        if torch.is_tensor(img):
            img_flat = img.numpy().flatten()
        else:
            img_flat = transforms.ToTensor()(img).numpy().flatten()
        target_features.append(img_flat)

    for img, _ in shadow_data:
        if torch.is_tensor(img):
            img_flat = img.numpy().flatten()
        else:
            img_flat = transforms.ToTensor()(img).numpy().flatten()
        shadow_features.append(img_flat)

    target_features = np.array(target_features)
    shadow_features = np.array(shadow_features)

    # Normalize features
    target_features = target_features / np.linalg.norm(target_features, axis=1)[:, np.newaxis]
    shadow_features = shadow_features / np.linalg.norm(shadow_features, axis=1)[:, np.newaxis]

    # Initialize lists for similar indices and k-th similarities
    similar_indices = []
    k_similarities = []

    # For each target image
    for i in range(len(target_data)):
        # Get current label
        current_label = target_labels[i]

        # Find shadow images with matching label
        matching_indices = np.where(shadow_labels == current_label)[0]

        if len(matching_indices) == 0:
            print(f"Warning: No shadow images found for label {current_label}")
            similar_indices.append(np.array([], dtype=int))
            k_similarities.append(None)
            continue

        # Get features for matching shadow images
        matching_features = shadow_features[matching_indices]

        if method == "cosine":
            similarities = cosine_similarity([target_features[i]], matching_features)[0]
        elif method == "l2":  # l2 distance
            similarities = -np.linalg.norm(target_features[i] - matching_features, axis=1)
        else:
            raise ValueError(f"Invalid method: {method}")

        # Find top-k most similar matching images
        k = min(topk, len(matching_indices))  # ensure k isn't larger than available matches
        top_k_indices = np.argsort(similarities)[-k:][::-1]  # sort in descending order
        similar_indices.append(matching_indices[top_k_indices])
        # Store k-th highest similarity
        k_similarities.append(similarities[top_k_indices[-1]])

    print(f"Found top-{topk} similar images with matching labels for {len(target_data)} target samples")

    # plot_pair_images(target_data, shadow_data, similar_indices, k, method)
    return similar_indices, target_labels, shadow_labels, k_similarities


def find_similar_images_clip(target_data, shadow_data, topk=5):
    """
    Find the top-k most similar shadow images for each target image using CLIP embeddings
    Args:
        target_data: list of target images [(image, label), ...]
        shadow_data: list of shadow images [(image, label), ...]
        topk: number of similar images to return for each target
    Returns:
        similar_indices: list of numpy arrays, each array contains indices of top-k most similar shadow images
        target_labels: numpy array of target labels
        shadow_labels: numpy array of shadow labels
        k_similarities: list of k-th highest similarity scores for each target
    """
    # Get labels
    target_labels = np.array([label for _, label in target_data])
    shadow_labels = np.array([label for _, label in shadow_data])
    # Get images
    target_images = [img for img, _ in target_data]
    shadow_images = [img for img, _ in shadow_data]

    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, device = get_clip_model()

    # Get CLIP features
    print("Computing CLIP embeddings for target images...")
    target_features = get_clip_features(target_images, model, preprocess, device)
    print("Computing CLIP embeddings for shadow images...")
    shadow_features = get_clip_features(shadow_images, model, preprocess, device)

    # Normalize features (CLIP features are already normalized, but just to be safe)
    target_features = target_features / np.linalg.norm(target_features, axis=1)[:, np.newaxis]
    shadow_features = shadow_features / np.linalg.norm(shadow_features, axis=1)[:, np.newaxis]

    # Initialize lists for similar indices and k-th similarities
    similar_indices = []
    k_similarities = []

    print("Finding similar images with matching labels...")
    # For each target image
    for i in range(len(target_data)):
        # Get current label
        current_label = target_labels[i]

        # Find shadow images with matching label
        matching_indices = np.where(shadow_labels == current_label)[0]

        if len(matching_indices) == 0:
            print(f"Warning: No shadow images found for label {current_label}")
            similar_indices.append(np.array([], dtype=int))
            k_similarities.append(None)
            continue

        # Get features for matching shadow images
        matching_features = shadow_features[matching_indices]

        # Calculate cosine similarities
        similarities = cosine_similarity([target_features[i]], matching_features)[0]

        # Find top-k most similar matching images
        k = min(topk, len(matching_indices))  # ensure k isn't larger than available matches
        top_k_indices = np.argsort(similarities)[-k:][::-1]  # sort in descending order
        similar_indices.append(matching_indices[top_k_indices])
        # Store k-th highest similarity
        k_similarities.append(similarities[top_k_indices[-1]])

    print(f"Found top-{topk} similar images with matching labels for {len(target_data)} target samples")

    # Plot example pairs using the helper function
    # plot_pair_images(target_data, shadow_data, similar_indices, topk, method="clip")
    return similar_indices, target_labels, shadow_labels, k_similarities


def find_similar_images_faiss(target_ds, shadow_ds, topk=10):
    """
    Find similar images using FAISS for efficient similarity search, filtering by matching labels.
    
    Args:
        target_ds: Target dataset
        shadow_ds: Shadow dataset
        topk: Number of similar images to find
        
    Returns:
        similar_indices: list of numpy arrays, each array contains indices of top-k most similar shadow images
        target_labels: numpy array of target labels
        shadow_labels: numpy array of shadow labels
        k_similarities: list of k-th highest similarity scores for each target
    """
    # Get labels
    target_labels = np.array([label for _, label in target_ds])
    shadow_labels = np.array([label for _, label in shadow_ds])
    
    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess, device = get_clip_model()
    
    # Process shadow dataset
    shadow_images = [img for img, _ in shadow_ds]
    
    print("Computing CLIP embeddings for shadow images...")
    shadow_features = get_clip_features(shadow_images, model, preprocess, device)
    
    # Normalize features for cosine similarity
    shadow_features = shadow_features / np.linalg.norm(shadow_features, axis=1)[:, np.newaxis]
    
    # Process target dataset
    target_images = [img for img, _ in target_ds]
    
    print("Computing CLIP embeddings for target images...")
    target_features = get_clip_features(target_images, model, preprocess, device)
    
    # Normalize features for cosine similarity
    target_features = target_features / np.linalg.norm(target_features, axis=1)[:, np.newaxis]
    
    # Create separate indices for each class
    print("Creating FAISS indices for each class...")
    class_indices = {}  # Dictionary to store indices for each class
    class_feature_maps = {}  # Dictionary to map local indices to global indices
    
    # Get unique classes
    unique_classes = np.unique(shadow_labels)
    
    # Create an index for each class
    for class_label in unique_classes:
        # Find shadow images with this class
        class_indices_list = np.where(shadow_labels == class_label)[0]
        
        if len(class_indices_list) == 0:
            continue
            
        # Get features for this class
        class_features = shadow_features[class_indices_list]
        
        # Create index for this class
        dimension = class_features.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Use Inner Product for cosine similarity
        
        # Add features to the index
        index.add(class_features)
        
        # Store the index and the mapping from local to global indices
        class_indices[class_label] = index
        class_feature_maps[class_label] = class_indices_list
    
    # Initialize lists for similar indices and k-th similarities
    similar_indices = []
    k_similarities = []
    
    print("Finding similar images with matching labels using FAISS...")
    import time
    start_time = time.time()
    
    # For each target image
    for i in range(len(target_ds)):
        # Get current label
        current_label = target_labels[i]
        
        # Check if we have an index for this class
        if current_label not in class_indices:
            print(f"Warning: No shadow images found for label {current_label}")
            similar_indices.append(np.array([], dtype=int))
            k_similarities.append(None)
            continue
        
        # Get the index for this class
        index = class_indices[current_label]
        feature_map = class_feature_maps[current_label]
        
        # Search for similar images
        k = min(topk, len(feature_map))  # ensure k isn't larger than available matches
        similarities, local_indices = index.search(target_features[i:i+1], k)
        
        # Convert local indices to global indices
        global_indices = feature_map[local_indices[0]]
        similar_indices.append(global_indices)
        
        # Store k-th highest similarity
        k_similarities.append(similarities[0][-1])
    
    end_time = time.time()
    print(f"Time taken for FAISS search: {end_time - start_time} seconds")
    print(f"Found top-{topk} similar images with matching labels for {len(target_ds)} target samples")
    
    return similar_indices, target_labels, shadow_labels, k_similarities
