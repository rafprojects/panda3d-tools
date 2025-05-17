import torch
import numpy as np
from PIL import Image
import open3d as o3d

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = "cuda" if torch.cuda.is_available() else "cpu"
midas.to(device)
midas.eval()

# Load and preprocess the input image
img = Image.open("ring.png")
if img.mode == 'RGBA':
    img = img.convert('RGB')  # Ensure image is RGB
img_np = np.array(img)
height, width, _ = img_np.shape
   
transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
input_batch = transform(img_np).to(device)

# Predict depth map
with torch.no_grad():
    prediction = midas(input_batch)

# Resize depth map to original image size
original_size = img.size[::-1]  # (height, width)
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=original_size,
    mode="bicubic",
    align_corners=False,
).squeeze()
disparity = prediction.cpu().numpy()

# Convert disparity to depth (0-10 meters range)
d_min = disparity.min()
d_max = disparity.max()
depth_map = (d_max - disparity) / (d_max - d_min) * 10  # Invert disparity to depth
depth_map = depth_map.astype(np.float32)

# Create RGBD image using Open3D
# color_image = o3d.io.read_image("input.jpg")
color_image = o3d.geometry.Image(img_np) 
depth_image = o3d.geometry.Image(depth_map)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_image, depth_image, convert_rgb_to_intensity=False
)

# Define camera intrinsics with default values
# width = color_image.get_width()
# height = color_image.get_height()
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width, height, fx=528, fy=528, cx=width / 2, cy=height / 2
)

# Generate point cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# Estimate normals for the point cloud
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Reconstruct mesh using Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Save the resulting mesh
o3d.io.write_triangle_mesh("output.obj", mesh)

print("3D model has been saved as 'output.obj'")