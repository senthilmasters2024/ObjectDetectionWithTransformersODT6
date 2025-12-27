import torch
from models.model import RGBDModel

model = RGBDModel()
model.eval()

rgb = torch.randn(1, 3, 512, 512)
depth = torch.randn(1, 1, 512, 512)

with torch.no_grad():
    outputs = model(rgb, depth)

rgb_feats = outputs["rgb"]
depth_feats = outputs["depth"]

print("RGB levels:", len(rgb_feats))
print("Depth levels:", len(depth_feats))

for i in range(len(rgb_feats)):
    print(f"Level {i}: RGB {rgb_feats[i].shape} | Depth {depth_feats[i].shape}")
