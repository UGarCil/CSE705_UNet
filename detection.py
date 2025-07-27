'''
This script is used for inference and visualization of a UNet model trained on a dataset.
It includes functions to visualize the model's predictions and generate confusion matrices.
'''
# 1. Standard library imports
import random
import os 
from os.path import join as jn
# 2. Related third party imports
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import tifffile as tiff
# 3. Local application/library specific imports
from UNetDataset import UNetDataset
from UNet import UNet
from torch.nn import DataParallel


# predicted/Truth       TRUE    FALSE
#       TRUE           truePOS  falsePOS
#       FALSE          falseNEG trueNEG

true_POS = 0
false_POS = 0
false_NEG = 0
true_NEG = 0
total = 0


def generateConfusionMatrix(truth:np.array,pred:np.array):
    '''
    Generates a confusion matrix image from the truth and predicted masks.
    Args:
        truth (np.array): The ground truth binary mask.
        pred (np.array): The predicted binary mask.
    Returns:
        np.array: An image representing the confusion matrix.
    '''
    #  binary masks of shape (512,512,1)
    # Convert them to boolean arrays
    truth_bool = truth.astype(bool)
    predicted_bool = pred.astype(bool)

    # Initialize the result array with zeros (black color)
    result = np.zeros((512, 512, 3), dtype=np.uint8)

    # Apply the conditions and assign colors
    # White: truth is True and predicted is True
    white_mask = (truth_bool & predicted_bool).squeeze()
    result[white_mask] = [255, 255, 255]  # white

    # Orange: truth is True and predicted is False
    orange_mask = (truth_bool & ~predicted_bool).squeeze()
    result[orange_mask] = [255, 165, 0]   # orange

    # Red: truth is False and predicted is True
    red_mask = (~truth_bool & predicted_bool).squeeze()
    result[red_mask] = [255, 0, 0]     # red
    
    black_mask = (~truth_bool & ~predicted_bool).squeeze()
    
    # Quantify each category
    white_count = np.sum(white_mask)
    orange_count = np.sum(orange_mask)
    red_count = np.sum(red_mask)
    black_count = np.sum(black_mask)
    total = white_count + orange_count + red_count + black_count
    counts = {
        "white": white_count,
        "orange": orange_count,
        "red": red_count,
        "black": black_count,
        "total":total
    }

    return result,counts




def single_image_inference(img_tensor, mask_tensor, device, isPlot=True, model=None):
    """
    Args:
        img_tensor (Tensor): Tensor of shape (3, H, W), already transformed
        mask_tensor (Tensor): Tensor of shape (1, H, W) or (H, W), already transformed
        device (str): "cuda" or "cpu"
        isPlot (bool): Whether to show the plots
        model (nn.Module): The model to use for inference
    Returns:
        dict: Dictionary of mask category counts
    """

    if model is None:
        raise ValueError("You must provide a model for inference.")

    img = img_tensor.float().to(device).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        pred_mask = model(img)

    # Prepare original image for display
    img_disp = img.squeeze(0).cpu().permute(1, 2, 0)  # (H, W, 3)

    # Prepare predicted mask
    pred_mask = pred_mask.squeeze(0).cpu().permute(1, 2, 0)  # (H, W, 1)
    pred_mask[pred_mask < 0] = 0
    pred_mask[pred_mask > 0] = 1
    pred_mask_np = pred_mask.numpy()

    # Prepare ground truth mask
    if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
        real_mask_np = mask_tensor.squeeze(0).numpy()  # (H, W)
    else:
        real_mask_np = mask_tensor.numpy()  # (H, W)
    real_mask_np = real_mask_np / real_mask_np.max()  # Normalize if needed
    real_mask_np = np.reshape(real_mask_np, (real_mask_np.shape[0], real_mask_np.shape[1], 1))  # (H, W, 1)

    # Generate confusion matrix
    confusion_matrix_image, mask_count = generateConfusionMatrix(truth=real_mask_np, pred=pred_mask_np)

    # Optional plot
    if isPlot:
        fig = plt.figure(figsize=(12, 3))
        titles = ["Original", "Predicted", "Truth", "Confusion Mask"]
        images = [img_disp, pred_mask.squeeze(-1), real_mask_np.squeeze(-1), confusion_matrix_image]

        for i in range(4):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(images[i], cmap="gray" if i != 3 else None)
            ax.set_title(titles[i])
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return mask_count





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    MODEL_PATH = "../trained_models/July21_2025/UNET_model_epoch_50_of_250.pth"
    MASKS_PATH = "../ISBI-2012-challenge/test-labels.tif"
    IMAGES_PATH = "../ISBI-2012-challenge/test-volume.tif"

    # Load full volumes
    volume = tiff.imread(IMAGES_PATH)  # shape: (N, H, W)
    labels = tiff.imread(MASKS_PATH)   # shape: (N, H, W)
    assert volume.shape[0] == labels.shape[0], "Mismatched number of slices"

    # Transformations (same as in your __init__)
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    label_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    # Load model
    # DD. MODEL
    # model = UNet()
    model = UNet(in_channels=3,num_classes=1).to(device)

    # Use DataParallel to spread the model across multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    

    # Inference on 3 random slices and plot them
    slice_indices = random.sample(range(volume.shape[0]), 3)
    for idx in slice_indices:
        img = image_transform(volume[idx])
        mask = label_transform(labels[idx])

        # counts = single_image_inference(img, mask, device)
        counts = single_image_inference(img, mask, device, isPlot=True, model=model)

        counts = {k: round(v, 3) for k, v in counts.items()}
        print(f"Slice {idx}: {counts}")

    # Full confusion matrix across all slices
    true_POS = false_POS = false_NEG = true_NEG = total = 0
    indices = list(range(volume.shape[0]))
    random.shuffle(indices)

    for counter, idx in enumerate(indices):
        img = image_transform(volume[idx])
        mask = label_transform(labels[idx])

        counts = single_image_inference(img, mask, device, isPlot=False, model=model)
        true_POS += counts["white"]
        false_POS += counts["red"]
        false_NEG += counts["orange"]
        true_NEG += counts["black"]
        total += counts["total"]

        if counter % 100 == 0 or counter == len(indices) - 1:
            final_counts = {
                "white": round(true_POS / total, 4),
                "red": round(false_POS / total, 4),
                "orange": round(false_NEG / total, 4),
                "black": round(true_NEG / total, 4),
                "TOTAL": round(total, 4)
            }
            print(f"After {counter} slices:", final_counts)
