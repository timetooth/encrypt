import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def get_transform():
    return T.Compose([
        T.Resize((512, 512)), # add a resize if required
        T.ToTensor(),
    ])

class MimicDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        if isinstance(df,str):
            if not os.path.exists(df): 
                raise Exception(f"Dataframe file not found: {df}")
            self.img_df = pd.read_csv(df)

        elif isinstance(df, pd.DataFrame):
            self.img_df = df
        
        else:
            raise Exception(f"Invalid type for dataframe: {type(df)}. It must be a file path or a pandas DataFrame.")

        self.transform = transform if transform is not None else get_transform()        

        if self.img_df.empty:
            raise Exception(f"Image dataframe is empty: {df}")
        
        required_columns = ["img_path", "folder_path", "dicom_id", "view_position"]
        for col in required_columns:
            if col not in self.img_df.columns: raise Exception(f"Image dataframe must contain '{col}' column")
        
        return

    def __len__(self):
        return len(self.img_df)
    
    def __getitem__(self, index):
        path = self.img_df.iloc[index]["img_path"]

        try:
            img = Image.open(path).convert("L")
        except Exception as e:
            raise Exception(f"Error loading image: {path}") from e

        if self.transform is not None:
            img = self.transform(img)
        
        return img
    


if __name__ == "__main__":
    import torchvision.utils as vutils
    from torch.utils.data import DataLoader

    dataset = MimicDataset("./dataset/mimic-cxr-dataset/image_paths_PA.csv")
    data = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)

    print(f"Total images: {len(data)}")
    
    batch = next(iter(data))
    print(f"Batch shape: {batch.shape}")
    vutils.save_image(batch, "./dataset/sample/sample_images.png", nrow=5, normalize=True)