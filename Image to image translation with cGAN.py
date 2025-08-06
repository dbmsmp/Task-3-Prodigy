import torch
from PIL import Image
from torchvision import transforms

img = Image.open("morning_mountain.jpg").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

x = transform(img).unsqueeze(0)

model = torch.load("pix2pix_mountain_day2night.pth", map_location=torch.device("cpu"))
model.eval()

with torch.no_grad():
    y = model(x)

output = y.squeeze(0).detach().cpu()
output = (output + 1) / 2

out_img = transforms.ToPILImage()(output.clamp(0, 1))
out_img.save("night_mountain.jpg")

print("✅ Night image saved as 'night_mountain.jpg'")
