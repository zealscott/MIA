import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import clip
from PIL import Image


def plot_image(img_tensor):
    """Helper function to properly display image tensor"""
    if torch.is_tensor(img_tensor):
        img = img_tensor.numpy()
    else:
        img = np.array(img_tensor)

    # Handle different tensor formats
    if img.shape[0] in [1, 3]:  # If channel-first format
        img = np.transpose(img, (1, 2, 0))

    # If grayscale, squeeze the channel dimension
    if img.shape[-1] == 1:
        img = img.squeeze()

    return img


def get_clip_features(images, model, preprocess, device, batch_size=128):
    """
    Get CLIP embeddings for a list of images in batches
    Args:
        images: list of (image, label) pairs
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: torch device
        batch_size: number of images to process at once
    Returns:
        features: numpy array of embeddings
    """
    all_features = []
    total_images = len(images)

    for i in range(0, total_images, batch_size):
        # print(f"Processing batch {i//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
        # Get current batch
        batch_end = min(i + batch_size, total_images)
        batch_images = images[i:batch_end]

        # Process batch
        processed_images = []
        for img in batch_images:
            if torch.is_tensor(img):
                img = transforms.ToPILImage()(img)
            processed_images.append(preprocess(img))

        # Stack batch and move to device
        image_batch = torch.stack(processed_images).to(device)

        # Get features and immediately move to CPU
        with torch.no_grad():
            features = model.encode_image(image_batch)
            features = features.cpu().numpy()

        all_features.append(features)

        # Clear GPU memory
        del image_batch, features
        torch.cuda.empty_cache()

    # Concatenate all batches
    return np.concatenate(all_features, axis=0)


def get_clip_model():
    """
    Load CLIP model and preprocessing
    """
    # If multiple GPUs available, use the first one
    if torch.cuda.device_count() > 1:
        device = f"cuda:0"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def plot_pair_images(target_data, shadow_data, similar_indices, topk=5, method="cosine"):
    """
    Plot a pair of target and shadow images
    """
    # plot two random target images and their most similar shadow images
    import matplotlib.pyplot as plt
    import random

    fig, axes = plt.subplots(2, topk + 1, figsize=(5 * (topk + 1), 8))
    for i in range(2):
        idx = i
        # Plot target image
        target_img = plot_image(target_data[idx][0])
        axes[i, 0].imshow(target_img, cmap="gray" if len(target_img.shape) == 2 else None)
        axes[i, 0].set_title(f"Target Image {idx}")
        axes[i, 0].axis("off")

        # Plot top-k similar images
        cur_similar = similar_indices[idx]
        for j in range(min(topk, len(cur_similar))):
            sim_idx = cur_similar[j]
            shadow_img = plot_image(shadow_data[sim_idx][0])
            axes[i, j + 1].imshow(shadow_img, cmap="gray" if len(shadow_img.shape) == 2 else None)
            axes[i, j + 1].set_title(f"Similar {j+1}")
            axes[i, j + 1].axis("off")

        # Fill empty slots if fewer than topk matches
        for j in range(len(cur_similar), topk):
            axes[i, j + 1].set_title("No match")
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"target_and_similar_images_{method}.png")
    plt.close()
