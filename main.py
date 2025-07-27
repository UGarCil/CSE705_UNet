'''
Auth: Uriel Garcilazo Cruz
      Snigdha Paney
Date created: 2025-06-20 (June 20, 2025)


Description:
This script represent the CONTROL module of the UNet architecture. It coordinates 
the loading of datasets, the initialization of the model, training and visualization of results.

Use of PEP8 conventions throughout the document
'''

# MODULES
# 1. Standard library imports
from os.path import join as jn
import os
# 2. Related third party imports
import torch
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# 3. Local application/library specific imports
from UNetDataset import UNetDataset
from UNet import UNet
from UNetVgg19_4downsamples import UNetVgg19, load_vgg19_weights



'''NOMENCLATURE
Throughout the document, the following symbols and their definitions are implemented:
- DD. %_NAME    Data definition. Followed by a description of the data, type and examples of implementation
- FD. %_NAME    Function definition Useful header that declares the definition of a function
- CD. %_NAME    Class definition. Useful header that declares the definition of a class

Whenever possible, a template for data structures is provided, with the following format:
example = {"A":str, "B":int, "C":[float, float, float]}

TEMPLATE FOR DD. %_NAME
fn_for_example (example))
... example["A"]
... example["B"]
for c in example["A"]:
    ... c

Whenever required, a function especially designed to consume the data structure will 
follow a naming convention that provides traceability to the data structure it consumes.

For example, fn_for_example(example) will consume the data structure example.


Algorithm:
1. Load the training and validation datasets using UNetDataset (DATALOADER MODULE)
2. Initialize the UNet model with the specified input channels and number of classes (MODEL MODULE)
    Define the optimizer (AdamW) and the loss function (BCEWithLogitsLoss).
4. For each epoch: (TRAINER MODULE)
    a. For each batch, perform a forward pass, compute the loss, and update the model parameters.
    b. Set the model to evaluation mode and iterate through the validation dataloader.
    c. Compute the validation loss without updating the model parameters.
    d. Print the training and validation losses for each epoch.
    e. Save the model state after each epoch.
5. 
'''

####################### (TRAINER MODULE) #######################
def trainer(model, train_dataloader, val_dataloader, optimizer, criterion, device, output_dir="output"):
    '''
    Trainer function to train the UNet model.
    Args:
        model (nn.Module): The UNet model to train.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function to compute the loss.
        device (str): Device to run the model on ('cuda' or 'cpu').
    Returns:
        None
    '''
    # if output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # initialize empty array to place losses
    train_losses = []
    val_losses = []
    # set model to training mode
    for epoch in range(EPOCHS):
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            y_pred = model(img)

            optimizer.zero_grad()
            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()
            val_loss = val_running_loss / (idx + 1)
            val_losses.append(val_loss)

        # Logging
        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)

        # Save model checkpoint every 10 epochs (or last one)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            torch.save(model.state_dict(), os.path.join(output_dir, f"UNET_model_epoch_{epoch + 1}_of_{EPOCHS}.pth"))
    print("Training complete.")
    return train_losses, val_losses

####################### (VISUALIZATION MODULE) #######################
def visualize_results(train_losses, val_losses, output_path="output"):
    '''
    Function to visualize training and validation losses.
    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    Returns:
        None
    '''
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(os.path.join(output_path, "loss_plot.png"))

##################### (HYPERPARAMETER EXECUTION MODULE) #############################

def hyper_wrapper(learning_rate=3e-4, batch_size=2, epochs=150, augmentation=True, optimizer_type='AdamW', vanilla=True):
    '''
    Main function to execute the training process under specified parameters.
    Its main function is to orchestrate the flow of information throught the Methodology indicated in the Detailed Report.
    Args:   
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for the DataLoader.
        epochs (int): Number of epochs to train the model.
        augmentation (bool): Whether to apply data augmentation.
        optimizer_type (str): Type of optimizer to use ('AdamW', 'SGD', 'Adam').
        vanilla (bool): If True, use a vanilla UNet without VGG19 backbone.
    Returns:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    '''
    # Set device and constants
    PATH = r"../ISBI-2012-challenge"
    PATH_IMAGES_TRAIN = jn(PATH,"train-volume.tif")
    PATH_MASKS_TRAIN = jn(PATH,"train-labels.tif")
    PATH_IMAGES_TEST = jn(PATH,"test-volume.tif")
    PATH_MASKS_TEST = jn(PATH,"test-labels.tif")
    LEARNING_RATE = learning_rate
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    AUGMENTATION = augmentation
    output_path = f"output_{EPOCHS}epochs_{'aug' if AUGMENTATION else 'noaug'}_{BATCH_SIZE}batch_{LEARNING_RATE}lr_{optimizer_type}_{'vanilla' if vanilla else 'vgg19'}"

    device = "cuda" if torch.cuda.is_available() else "cpu"


        
    ####################### (DATA LOADER MODULE) ####################### 
    # DD. UNET_DATASET 
    # dataset = UNetDataset.UNetDataset() <local>
    # interp. dataset incorporated as an object, and modified by transformations
    # in resolution and datatypes
    # %_dataset = UNetDataset(PATH_IMAGES,PATH_MASKS)
    train_dataset = UNetDataset(PATH_IMAGES_TRAIN,PATH_MASKS_TRAIN, augment=AUGMENTATION)
    val_dataset = UNetDataset(PATH_IMAGES_TEST,PATH_MASKS_TEST, augment=False)

    # DD. %_DATALOADER
    # %_dataloader = DataLoader()
    # interp. an object optimized for pytorch architecture, representing a train or validation dataset
    # generator = torch.Generator().manual_seed(42)
    ## Calculate lengths
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size = BATCH_SIZE,
                                shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size = BATCH_SIZE,
                                shuffle=False)

    ####################### (MODEL MODULE) #######################
    # DD. MODEL
    # model = UNet()
    # interp. a UNet model with a specified number of input channels and output classes
    # model is one of:
    # - vanilla UNet
    # - UNet with VGG19 backbone
    if vanilla:
        model = UNet(in_channels=3, num_classes=1).to(device)
    else:
        vgg = load_vgg19_weights()
        model = UNetVgg19(in_channels=3, num_classes=1, vgg19=vgg).to(device)

    # Use DataParallel to spread the model across multiple GPUs if available.
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)


    # DD. OPTIMIZER
    # optimizer = optim.AdamW()
    # interp. technique to incorporate backpropagation
    # optiizer is one of:
    # - AdamW
    # - SGD
    # - Adam
    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
    # DD. CRITERION
    # criterion = nn.BCEWithLogitsLoss()
    # interp. loss function
    criterion = nn.BCEWithLogitsLoss()
    # Training the model
    train_losses, val_losses = trainer(model, train_dataloader, val_dataloader, optimizer, criterion, device, output_dir=output_path)

    # Visualize results
    visualize_results(train_losses, val_losses, output_path=output_path)

    return train_losses, val_losses

####################### MAIN EXECUTION MODULE ######################
'''
This section of the code orchestrates the Methodology described in the Detailed Report. It systematically performs the following operations:
- Traverse the hyperparameter space defined by the 4 different learning rates and 4 different batch sizes.
    - Train the network with the specified hyperparameters, saving the model every 10 epochs and collecting loss values for validation adn training.
    - Store the results in a matrix, where each cell corresponds to a specific combination of learning rate and batch size.
- Evaluate the performance of different optimizers (AdamW, SGD, Adam) using the best hyperparameters found in the previous step.
- Run the UNet architecture with and without a VGG19 backbone, comparing their performance.

'''



learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
batch_sizes = [1, 2, 4, 8]
EPOCHS = 100

results = np.zeros((len(batch_sizes), len(learning_rates)))

for i, bs in enumerate(batch_sizes):
    for j, lr in enumerate(learning_rates):
        print(f"Evaluating model with LR={lr}, BS={bs}")
        log_path = f"logs/losses_lr{lr}_bs{bs}.npy"
        # Save loss data to a log file, and if we already made one, then there's no point on retraining it
        if os.path.exists(log_path):
            print(f"Found cached result at {log_path}, loading...")
            losses = np.load(log_path, allow_pickle=True).item()
            train_losses, val_losses = losses["train"], losses["val"]
        else:
            # Train the model with the specified hyperparameters
            print(f"Training model from scratch...")
            train_losses, val_losses = hyper_wrapper(
                learning_rate=lr,
                batch_size=bs,
                epochs=EPOCHS,
                augmentation=True
            )
            os.makedirs("logs", exist_ok=True)
            np.save(log_path, {"train": train_losses, "val": val_losses})

        # Store final val loss in results matrix
        results[i, j] = val_losses[-1]


# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    results,
    annot=True,
    fmt=".4f",
    xticklabels=learning_rates,
    yticklabels=batch_sizes,
    cmap=sns.color_palette("Blues", as_cmap=True),
    linewidths=0.5,
    linecolor='gray'
)
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.title("Validation Loss Heatmap (White = Low Loss, Marine Blue = High)")
plt.tight_layout()
plt.savefig("validation_loss_heatmap.png")
plt.show()


####### Evaluate optimizer performance ##############
optimizer_types = ['AdamW', 'SGD', 'Adam']
best_lr = 0.0001
best_bs = 1
EPOCHS = 100

optimizer_results = {}

for opt in optimizer_types:
    print(f"\nRunning with optimizer = {opt}")
    log_path = f"logs/losses_lr{best_lr}_bs{best_bs}_opt{opt}.npy"
    # If data is already cached, load it
    if os.path.exists(log_path):
        print(f"Found cached result at {log_path}, loading...")
        losses = np.load(log_path, allow_pickle=True).item()
        train_losses, val_losses = losses["train"], losses["val"]
    else:
        # Train the model with the specified optimizer
        print(f"Training model from scratch with optimizer = {opt}...")
        train_losses, val_losses = hyper_wrapper(
            learning_rate=best_lr,
            batch_size=best_bs,
            epochs=EPOCHS,
            augmentation=True,
            optimizer_type=opt
        )
        os.makedirs("logs", exist_ok=True)
        np.save(log_path, {"train": train_losses, "val": val_losses})

    optimizer_results[opt] = val_losses

# Plot validation loss over time for each optimizer
plt.figure(figsize=(10, 5))
for opt in optimizer_types:
    plt.plot(optimizer_results[opt], label=opt)

plt.title(f"Optimizer Comparison (LR={best_lr}, BS={best_bs})")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("optimizer_comparison_plot.png")
plt.show()


########## Compare UNetVgg19 architecture against vanilla UNet ##########


# Use best hyperparameters found earlier
best_lr = 0.0001
best_bs = 1
optim_type = "AdamW"

arch_results = {}

# --- RAW UNET ---
print(f"\nRunning with raw UNet (no pretrained backbone)")
log_path_raw = f"logs/losses_unet_raw_lr{best_lr}_bs{best_bs}.npy"
# Search for cached data first. If can't find it, then train.
if os.path.exists(log_path_raw):
    print(f"Found cached result at {log_path_raw}, loading...")
    losses = np.load(log_path_raw, allow_pickle=True).item()
    train_losses_raw, val_losses_raw = losses["train"], losses["val"]
else:
    print(f"Training raw UNet from scratch...")
    train_losses_raw, val_losses_raw = hyper_wrapper(
        learning_rate=best_lr,
        batch_size=best_bs,
        epochs=EPOCHS,
        vanilla=True,
        optimizer_type= optim_type
    )
    np.save(log_path_raw, {"train": train_losses_raw, "val": val_losses_raw})

arch_results["UNet"] = {"train": train_losses_raw, "val": val_losses_raw}

# --- UNET + VGG19 ---
print(f"\nRunning with UNet + VGG19 pretrained encoder")
log_path_vgg = f"logs/losses_unet_vgg19_lr{best_lr}_bs{best_bs}.npy"

if os.path.exists(log_path_vgg):
    print(f"Found cached result at {log_path_vgg}, loading...")
    losses = np.load(log_path_vgg, allow_pickle=True).item()
    train_losses_vgg, val_losses_vgg = losses["train"], losses["val"]
else:
    print(f"Training UNet with VGG19 backbone...")
    train_losses_vgg, val_losses_vgg = hyper_wrapper(
        learning_rate=best_lr,
        batch_size=best_bs,
        epochs=EPOCHS,
        vanilla=False,
        optimizer_type= optim_type
    )
    np.save(log_path_vgg, {"train": train_losses_vgg, "val": val_losses_vgg})

arch_results["UNetVgg19"] = {"train": train_losses_vgg, "val": val_losses_vgg}

#### Plot the train and validation losses, comparing the performance between the two networks
# Plot Training Loss
plt.figure(figsize=(10, 5))
# trim the vertical axis to 1.5
plt.ylim(0, 1.5)  
for model_name in arch_results:
    plt.plot(arch_results[model_name]["train"], label=model_name)
plt.title("Training Loss: UNet vs UNet + VGG19")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("unet_vs_vgg19_train_loss.png")
plt.show()

# Plot Validation Loss
plt.figure(figsize=(10, 5))
# trim the vertical axis to 1.5
plt.ylim(0, 1.5)  
for model_name in arch_results:
    plt.plot(arch_results[model_name]["val"], label=model_name)
plt.title("Validation Loss: UNet vs UNet + VGG19")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("unet_vs_vgg19_val_loss.png")
plt.show()
