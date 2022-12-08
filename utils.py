import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import *


MODEL_TYPES = ['Ensemble', 'Uniform Soup', 'Weighted Soup', 
            'Greedy Soup', 'Shephard Soup']

def plt_all_model_accs(val_accs):
    names = [f'Model {i+1}' for i in range(len(val_accs))]
    assert len(names) == len(val_accs)
    
    i = np.argmax(np.array(val_accs))
    colors = ['blue'] * len(val_accs)
    colors[i] = 'green'

    plt.bar(names, np.array(val_accs), color=colors)
    plt.title("Validation Accuracy By Model")
    plt.ylim([0.75, 0.85])
    plt.xticks(rotation=30, ha='right')
    plt.show()

def plt_diff_model_accs(val_accs, model_type=''):
    names = ['Best Single Model']
    if model_type == 'Ensemble':
        names.extend(MODEL_TYPES[:1])
    elif model_type == 'Uniform Soup':
        names.extend(MODEL_TYPES[:2])
    elif model_type == 'Weighted Soup':
        names.extend(MODEL_TYPES[:3])
    elif model_type == 'Greedy Soup':
        names.extend(MODEL_TYPES[:4])
    elif model_type == 'Shephard Soup':
        names.extend(MODEL_TYPES)

    assert len(names) == len(val_accs)
    
    colors = ['blue'] * (len(val_accs) - 1) + ['green']
    
    plt.bar(names, np.array(val_accs), color=colors)
    plt.title("Validation Accuracy By Model")
    plt.ylim([0.75, 0.85])
    plt.xticks(rotation=30, ha='right')
    plt.show()

##############################################  
########## Dataset Helper Functions ##########    
##############################################  

BICUBIC = InterpolationMode.BICUBIC

def convert_image_to_rgb(image):
    return image.convert("RGB")

def transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def create_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset,
                                batch_size=128,
                                shuffle=False,
                                num_workers=2,
                                drop_last=False)
              

##############################################  
################## ARCHIVE ###################    
##############################################  


# Fine tunes k models with different hyperparmeters.
# Note: This function was used to create the 16 fine-tuned models 
# being used in the notebook. 
def create_k_models(k=2):
    models = []
    for i in range(k):
      # Load the pre-trained model
      model = get_model()
      
      # Fine Tune the head of the 
      fine_tuned_model = fine_tune_model(model)

      # Save the model parameters
      torch.save(model.state_dict(), f'model_{i}')

    return models