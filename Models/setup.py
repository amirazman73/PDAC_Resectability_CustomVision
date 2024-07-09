from Models.multi_channel_model import ViT_3D_Classifier

def setup_model():
    model = ViT_3D_Classifier(
        num_classes=3,        # Number of output classes
        patch_size=(5, 5, 5), # Size of each patch
        emb_size=64,          # Embedding size for each patch
        num_heads=4,          # Number of attention heads
        mlp_dim=128,          # Dimension of the MLP layer
        seq_len=64,           # Length of the sequence (number of patches)
        k=32,                 # Linformer projection dimension
        dropout=0.1           # Dropout rate
    )
    return model