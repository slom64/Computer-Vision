import pybullet as p
import pybullet_data
import time
import csv
import os
import numpy as np

from config import *
import engine

# Connect to PyBullet GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.8)

# Disable the noisy GUI panels
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.setPhysicsEngineParameter(numSolverIterations=20)

engine.initialize()

def world_to_ndc(pos, view_mat, proj_mat):
    V = np.array(view_mat).reshape((4,4), order='F')
    P = np.array(proj_mat).reshape((4,4), order='F')
    
    p_world = np.array([pos[0], pos[1], pos[2], 1.0])
    p_view = V.dot(p_world)
    p_clip = P.dot(p_view)
    
    if p_clip[3] <= 0:
        return None
        
    p_ndc = p_clip / p_clip[3]
    return p_ndc[0:3]

# Adjust camera to look at the center of the camera view
p.resetDebugVisualizerCamera(cameraDistance=BOX_SIZE*1.0, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0, 0, (CAM_Z_MAX+CAM_Z_MIN)/2])

# CSV Setup
os.makedirs('frames', exist_ok=True)
csv_file = open('dataset.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'pellet_id', 'is_agglomerate', 'x_center', 'y_center', 'width', 'height'])

frame_count = 0
physics_step_counter = 0
steps_per_record = int(1.0 / (RECORD_FPS * DT)) 

print("Simulation started. Writing to dataset.csv and capturing frames...")

# ---------------------------------------------------------
# AI CAMERA SETUP (Decoupled from the GUI window for speed)
# ---------------------------------------------------------
ai_view_mat = p.computeViewMatrix(
    cameraEyePosition=[0, BOX_SIZE*1.2, (CAM_Z_MAX+CAM_Z_MIN)/2],
    cameraTargetPosition=[0, 0, (CAM_Z_MAX+CAM_Z_MIN)/2],
    cameraUpVector=[0, 0, 1]
)
# Using Perspective projection for a more realistic ML dataset
ai_proj_mat = p.computeProjectionMatrixFOV(
    fov=60.0, 
    aspect=1.0, 
    nearVal=0.1, 
    farVal=100.0
)

try:
    while p.isConnected():
        update_text = (physics_step_counter % steps_per_record == 0)
        engine.update(update_text)
        physics_step_counter += 1
        
        if physics_step_counter % steps_per_record == 0:
            
            # Capture frame using the fast fixed camera setup (No hardware bugs, blazing fast at 512x512)
            img_arr = p.getCameraImage(IMG_WIDTH, IMG_HEIGHT, viewMatrix=ai_view_mat, projectionMatrix=ai_proj_mat)
            rgba = img_arr[2]
            
            try:
                from PIL import Image
                import numpy as np
                # PyBullet can return a tuple or numpy array depending on versions
                if isinstance(rgba, tuple):
                    np_img = np.array(rgba, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
                else:
                    np_img = rgba
                
                img = Image.fromarray(np_img, 'RGBA').convert('RGB')
                img.save(os.path.join('frames', f'frame_{frame_count}.jpg'))
            except Exception as e:
                print("Failed to save image:", e)
                
            yolo_lines = []
            
            for base_id, info in engine.all_pellets.items():
                visible = True
                
                u_min, v_min = 1.0, 1.0
                u_max, v_max = 0.0, 0.0
                
                # Check if EVERY sphere in this agglomerate is inside the camera margins using true perspective (NDC)
                for b in info['bodies']:
                    pos, _ = p.getBasePositionAndOrientation(b)
                    
                    ndc = world_to_ndc(pos, ai_view_mat, ai_proj_mat)
                    if ndc is None:
                        visible = False
                        break
                        
                    ndc_x, ndc_y, ndc_z = ndc[0], ndc[1], ndc[2]
                    
                    # -1 to 1 is the screen bounds. We enforce an inner margin (e.g. 5%)
                    if (ndc_x < -1.0 + NDC_MARGIN or ndc_x > 1.0 - NDC_MARGIN or
                        ndc_y < -1.0 + NDC_MARGIN or ndc_y > 1.0 - NDC_MARGIN or
                        ndc_z < -1.0 or ndc_z > 1.0):
                        visible = False
                        break
                        
                    # Calculate bounding box by projecting 6 surface points of the sphere
                    pts_world = [
                        [pos[0]+BASE_RADIUS, pos[1], pos[2]],
                        [pos[0]-BASE_RADIUS, pos[1], pos[2]],
                        [pos[0], pos[1]+BASE_RADIUS, pos[2]],
                        [pos[0], pos[1]-BASE_RADIUS, pos[2]],
                        [pos[0], pos[1], pos[2]+BASE_RADIUS],
                        [pos[0], pos[1], pos[2]-BASE_RADIUS]
                    ]
                    for pt in pts_world:
                        pt_ndc = world_to_ndc(pt, ai_view_mat, ai_proj_mat)
                        if pt_ndc is not None:
                            u = (pt_ndc[0] + 1) / 2.0
                            v = (1 - pt_ndc[1]) / 2.0
                            u_min = min(u_min, u)
                            u_max = max(u_max, u)
                            v_min = min(v_min, v)
                            v_max = max(v_max, v)
                        
                if visible:
                    x_center = (u_min + u_max) / 2.0
                    y_center = (v_min + v_max) / 2.0
                    width = (u_max - u_min)
                    height = (v_max - v_min)
                    
                    # Class 0 for all pellets (YOLO tracking)
                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    # Log for the Sequence Model Dataset
                    csv_writer.writerow([frame_count, info['id'], info['is_agglomerate'], x_center, y_center, width, height])
                    
            if len(yolo_lines) > 0:
                with open(os.path.join('frames', f'frame_{frame_count}.txt'), 'w') as f:
                    f.writelines(yolo_lines)
                    
            frame_count += 1
            csv_file.flush() # Ensure it writes during the simulation
except Exception as e:
    print("Simulation stopped:", e)
finally:
    csv_file.close()
    print("dataset.csv closed cleanly.")