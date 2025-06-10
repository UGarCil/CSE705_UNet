# MODULES
from constants import *
from UNetDataset import UNetDataset
from UNet import UNet
from torch import optim
from torch.nn import DataParallel
# DD

# DD. UNET_DATASET
# dataset = UNetDataset()
# interp. dataset incorporated as an object, and modified by transformations
# in resolution and datatypes
# dataset = UNetDataset(PATH_IMAGES,PATH_MASKS)
train_dataset = UNetDataset(PATH_IMAGES_TRAIN,PATH_MASKS_TRAIN)
val_dataset = UNetDataset(PATH_IMAGES_TEST,PATH_MASKS_TEST)

# DD. %_DATALOADER
# %_dataloader = DataLoader()
# interp. an object optimized for pytorch architecture, representing a train or validation dataset
# generator = torch.Generator().manual_seed(42)
## Calculate lengths
# total_length = len(dataset)
# train_length = int(total_length * 0.9)
# val_length = total_length - train_length
# train_dataset, val_dataset = random_split(dataset,[train_length,val_length],generator=generator)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle=False)
val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle=False)

# DD. MODEL
# model = UNet()
model = UNet(in_channels=3,num_classes=1).to(device)

# Use DataParallel to spread the model across multiple GPUs if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)


# DD. OPTIMIZER
# optimizer = optim.AdamW()
# interp. technique to incorporate backpropagation
optimizer = optim.AdamW(model.parameters(),lr=LEARNING_RATE)
# DD. CRITERION
# criterion = nn.BCEWithLogitsLoss()
# interp. loss function
criterion = nn.BCEWithLogitsLoss()


##################### CODE #############################
for epoch in tqdm(range(EPOCHS)):
    # train
    model.train()
    train_running_loss = 0
    for idx,img_mask in enumerate(tqdm(train_dataloader)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)
        y_pred = model(img)
        
        optimizer.zero_grad()
        
        loss = criterion(y_pred,mask)
        train_running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/idx+1
    # validation
    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for idx,img_mask in enumerate(tqdm(val_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            y_pred = model(img)
            loss = criterion(y_pred,mask)
            
            val_running_loss += loss.item()
        val_loss = val_running_loss/idx+1
    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print("-"*30)
    torch.save(model.state_dict(),f"UNET_model_{epoch}_{EPOCHS}.pth")