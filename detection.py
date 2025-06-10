import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random
from UNetDataset import UNetDataset
from UNet import UNet
import os 
from os.path import join as jn
import numpy as np




PATH_MASKS = "../../20x_Green_PROCESSED/masks"
PATH_IMAGES = "../../20x_Phase_PROCESSED/original"

# predicted/Truth       TRUE    FALSE
#       TRUE           truePOS  falsePOS
#       FALSE          falseNEG trueNEG

true_POS = 0
false_POS = 0
false_NEG = 0
true_NEG = 0
total = 0

def pred_show_image_grid(images_path,masks_path, model_pth, device):
    # Load the model with its previous parameters
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    # Create a dataset for UNet
    image_dataset = UNetDataset(images_path,masks_path)
    images = []
    orig_masks = []
    pred_masks = []

    counter = 0
    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0]=0
        pred_mask[pred_mask > 0]=1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

        counter += 1
        if counter > 3:
            break

    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3*len(images)+1):
       fig.add_subplot(3, len(images), i)
       plt.imshow(images[i-1], cmap="gray")
    plt.show()

def generateConfusionMatrix(truth:np.array,pred:np.array):
    # Assuming truth and predicted are your binary masks of shape (512,512,1)
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




def single_image_inference(image_pth, mask_pth, device, isPlot=True):
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = Image.open(image_pth).convert("RGB")
    img = transform(img).float().to(device)
    img = img.unsqueeze(0)
    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)
    # img_np = np.array(img)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1
    pred_mask_np = np.array(pred_mask)
    
    real_mask = Image.open(mask_pth).convert("L")
    real_mask_np = np.array(real_mask)/255.
    maskNpShape = real_mask_np.shape
    real_mask_np = np.reshape(real_mask_np,(maskNpShape[1],maskNpShape[0],1))
    confusion_matrix_image,mask_count = generateConfusionMatrix(truth=real_mask_np,pred=pred_mask_np)
    
    if isPlot:
        fig = plt.figure()
        for i in range(1, 5): 
            ax = fig.add_subplot(1, 4, i)
            ax.axis("off")
            if i == 1:
                plt.imshow(img, cmap="gray")
                ax.set_title("Original")
            elif i==2:
                plt.imshow(pred_mask, cmap="gray")
                ax.set_title("Predicted")
            elif i==3:
                plt.imshow(real_mask, cmap="gray")
                ax.set_title("Truth")
            else:
                plt.imshow(confusion_matrix_image)
                ax.set_title("Confusion Mask")
        plt.show()

    return mask_count

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "./trained_models/Aug_21/UNET_model_123_200.pth"
    IMAGES_PATH = r"C:\Users\Uriel\Desktop\fluorecence_Aug19_dataset\flourecence_20x\dataset\test\origs"
    MASKS_PATH = r"C:\Users\Uriel\Desktop\fluorecence_Aug19_dataset\flourecence_20x\dataset\test\masks"
    imagesinPATH = [jn(IMAGES_PATH,imgName) for imgName in os.listdir(IMAGES_PATH)]
    # imagesinMASK = sorted([jn(MASKS_PATH,imgName) for imgName in os.listdir(MASKS_PATH)])
    model = UNet(in_channels=3, num_classes=1).to(device)
    # print(model.parameters)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
    
    for img_path in random.sample(imagesinPATH,10):
        
        # SINGLE_IMG_PATH = imagesinPATH[i]
        SINGLE_IMG_MASK = img_path.replace("orig","mask")

        counts = (single_image_inference(img_path, SINGLE_IMG_MASK, device))
        counts = {k:round(v,3) for k,v in counts.items()}
        # print(counts)
        
        
    # calculate confusion matrix values
    counter = 0
    random.shuffle(imagesinPATH)
    for img_path in imagesinPATH:
        # get the name of the mask
        SINGLE_IMG_MASK = img_path.replace("orig","mask")
        # get the confusion masks, this time without plotting
        counts = single_image_inference(img_path, SINGLE_IMG_MASK, device,isPlot=False)
        
        true_POS += counts["white"]
        false_POS += counts["red"]
        false_NEG += counts["orange"]
        true_NEG += counts["black"]
        total += counts["total"]
        counter += 1
    # total = len(imagesinPATH)
        final_counts = {"white":round(true_POS/total,4),
                        "red":round(false_POS/total,4),
                        "orange":round(false_NEG/total,4),
                        "black":round(true_NEG/total,4),
                        "TOTAL":round(total,4)}
        if counter % 100:
            print(final_counts)
    