from fastsam import FastSAM, FastSAMPrompt
import os
import ipdb
import time
import torch
import cv2
import pyrealsense2 as rs
import trimesh
import open3d as o3d
import numpy as np
import supervision as sv

# Adding FoundationPose path to sys
import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

# FounbdationPose imports
from estimater import *
from Utils import *

# FoundationPose stuff
#####################################################
# Save the original `__init__` and `register` methods
original_init = FoundationPose.__init__
original_register = FoundationPose.register

# Modify `__init__` to add `is_register` attribute
def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None, debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_registered = False  # Initialize as False

# Modify `register` to set `is_register` to True when a pose is registered
def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_registered = True  # Set to True after registration
    return pose

# Apply the modifications
FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

# Convert masks to boolean (True/False)
def masks_to_bool(masks):
    if type(masks) == np.ndarray:
        return masks.astype(bool)
    return masks.cpu().numpy().astype(bool)

def annotate_image(image_path: str, masks: np.ndarray) -> np.ndarray:
    image = cv2.imread(image_path)

    xyxy = sv.mask_to_xyxy(masks=masks)
    detections = sv.Detections(xyxy=xyxy, mask=masks)

    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
    return mask_annotator.annotate(scene=image.copy(), detections=detections)

def depth_to_xyz_map(depth_image, K, uvs=None):
    invalid_mask = (depth_image<0.1)
    H,W = depth_image.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth_image[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map

def toOpen3dCloud(points,colors=None,normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if colors.max()>1:
            colors = colors/255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud

def compute_oriented_bounding_box(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_oriented_bounding_box()
    return obb

def filter_masks_by_dimension(obbs, masks, masks_bool, cad_dimensions, tolerance):
    # filtered_obbs = []
    # filtered_masks = []
    # filtered_masks_bool = []
    chosen_obb = None
    chosen_mask = None
    chosen_mask_bool = None
    min_diff = float('inf')
    for i, obb in enumerate(obbs):
        l, b, h = obb.extent
        print(f"l = {l}, b = {b}, h = {h}", f"cad_dimensions = {cad_dimensions}")
        if (abs(l - cad_dimensions[0])) < tolerance and (abs(b - cad_dimensions[1])) < tolerance:
            if abs(l - cad_dimensions[0]) + abs(b - cad_dimensions[1]) < min_diff:
                min_diff = abs(l - cad_dimensions[0]) + abs(b - cad_dimensions[1])
                chosen_obb = obb
                chosen_mask = masks[i]
                chosen_mask_bool = masks_bool[i]
            # filtered_obbs.append(obb)
            # filtered_masks.append(masks[i])
            # filtered_masks_bool.append(masks_bool[i])
    print("Min diff: ", min_diff)
    return chosen_obb, chosen_mask, chosen_mask_bool

def visualize_pose(image, bbox, intrinsics, center_pose, ):
    vis = draw_posed_3d_box(intrinsics, img=image, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)
    return vis

# Define device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE = {DEVICE}")

# Set paths
FAST_SAM_CHECKPOINT_PATH = "/home/shruthi/Documents/Code/FastSAM/weights/FastSAM.pt"
fast_sam = FastSAM(FAST_SAM_CHECKPOINT_PATH)
# IMAGE_PATH = "/home/shruthi/Documents/Code/ZeroShotPose/test_input_data/hardcore_bearing/rgb.png"
# DEPTH_PATH = "/home/shruthi/Documents/Code/ZeroShotPose/test_input_data/hardcore_bearing/depth_raw.npy"
# CAD_PATH = "/home/shruthi/Documents/Code/ZeroShotPose/test_input_data/hardcore_bearing/hardcore_bearing.stl"
CAD_PATH = "/home/shruthi/Documents/Code/ZeroShotPose/test_input_data/skateboard_base/skateboard_base_fusion.stl"

os.makedirs("fastsam_results", exist_ok=True)

# Load CAD model
mesh = trimesh.load(CAD_PATH)

# Center and scale the CAD model
AABB = mesh.bounds
center = np.mean(AABB, axis=0)
mesh.vertices -= center
mesh.apply_scale(0.001)

# Model object
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
pose_estimation_model = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    glctx=glctx
)

i = 0

pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
align_to = rs.stream.color
align = rs.align(align_to)

# Set exposure
# Get the sensor once at the beginning. (Sensor index: 1)
sensor = pipeline.get_active_profile().get_device().query_sensors()[1]

# Set the exposure anytime during the operation
sensor.set_option(rs.option.exposure, 150.000)

while True:
    if i % 10 == 0:
        pose_estimation_model.is_registered = False

    color = None
    depth = None

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    depth_frame = aligned_frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    if not pose_estimation_model.is_registered:
        cv2.imwrite("fastsam_results/color_image.png", color_image)
        IMAGE_PATH = "/home/shruthi/Documents/Code/ZeroShotPose/fastsam_results/color_image.png"

        # Run FastSAM and plot image
        results = fast_sam(
            source=IMAGE_PATH,
            device=DEVICE,
            retina_masks=True,
            imgsz=1024,
            conf=0.5,
            iou=0.6)
        prompt_process = FastSAMPrompt(IMAGE_PATH, results, device=DEVICE)
        masks = prompt_process.everything_prompt()
        masks_bool = masks_to_bool(masks)
        # annotated_image=annotate_image(image_path=IMAGE_PATH, masks=masks_bool)
        # sv.plot_image(image=annotated_image, size=(8, 8))

        # Create pointcloud from depth image and intrinsics
        # depth_image = np.load(DEPTH_PATH)
        # color_image = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        intrinsics = np.array(
            [[617.077, 0.0, 314.71], [0.0, 617.473, 241.377], [0.0, 0.0, 1.0]]
        )

        xyz_map = depth_to_xyz_map(depth_image, intrinsics)
        xyz_map_flattened = np.nan_to_num(xyz_map.astype(np.float64)).reshape(-1, 3)
        color_image_flattened = color_image.reshape(-1, 3)
        pcd = toOpen3dCloud(xyz_map_flattened, color_image_flattened)

        # Convert masks to 3D oriented bounding boxes
        geometries = [pcd]
        obbs = []
        for mask in masks_bool:
            mask_points = xyz_map[mask]
            mask_points = mask_points[mask_points[:, 2] > 0]  # Filter out points with zero depth
            obb = compute_oriented_bounding_box(mask_points)
            obbs.append(obb)
            geometries.append(obb)

        # o3d.visualization.draw_geometries(geometries)

        # Filter masks based on oriented bounding box's volume
        # TODO: Filter based on length and breadth alone - don't use volume or height
        cad_mesh = o3d.io.read_triangle_mesh(CAD_PATH)
        cad_dimensions = cad_mesh.get_oriented_bounding_box().extent
        chosen_obb, chosen_mask, chosen_mask_bool = filter_masks_by_dimension(obbs, masks, masks_bool, cad_dimensions, tolerance=50.0)

        # Visualize the point cloud with filtered oriented bounding boxes
        geometries = [pcd] + [chosen_obb]
        # o3d.visualization.draw_geometries(geometries)

        # image_with_selected_mask=annotate_image(image_path=IMAGE_PATH, masks=np.asanyarray([chosen_mask_bool]))
        # sv.plot_image(image=image_with_selected_mask, size=(8, 8))

    depth_image = depth_image * 0.001

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)

    # Get the bounding box of the mesh
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    vis_image = np.copy(color_image)

    intrinsics = np.array(
        [[617.077, 0.0, 314.71], [0.0, 617.473, 241.377], [0.0, 0.0, 1.0]]
    )

    visualization_image = np.copy(color_image)
    if pose_estimation_model.is_registered:
        pose = pose_estimation_model.track_one(rgb=color_image, depth=depth_image, K=intrinsics, iteration=2)
        center_pose = pose @ np.linalg.inv(to_origin)
        visualization_image = visualize_pose(visualization_image, bbox, intrinsics, center_pose)
    else:
        print("Running foundation pose registration")
        start_time = time.time()
        pose = pose_estimation_model.register(K=intrinsics, rgb=color_image, depth=depth_image, ob_mask=chosen_mask.cpu().numpy(), iteration=4)
        print(f"Time taken: {time.time() - start_time}")
        center_pose = pose @ np.linalg.inv(to_origin)
        visualization_image = visualize_pose(visualization_image, bbox, intrinsics, center_pose)
        
    cv2.imshow('Pose Estimation & Tracking', visualization_image[..., ::-1])
    cv2.waitKey(1)

    i += 1