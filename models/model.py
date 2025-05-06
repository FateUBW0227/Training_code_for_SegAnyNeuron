import torch

# from models.unet_3D import UNet3D
# from models.UnetModel4 import UNet3D
# from models.UnetModel4_withoutfeature import UNet3D
# from models.UnetModel4_withfeature2 import UNet3D

# -1 qu's model; 0 without feature; 7 with feature.
def LoadModel(model_config, modelPath=None, type=7):
    if type == -1:
        from models.UnetModel4 import UNet3D
    elif type == 0:
        from models.UnetModel4_withoutfeature import UNet3D
    elif type == 7:
        from models.UnetModel4_withfeature2 import UNet3D

    model = UNet3D(**model_config)
    if not modelPath is None:
        ckpt = torch.load(modelPath)
        model.load_state_dict(ckpt['state_dict'])
    return model