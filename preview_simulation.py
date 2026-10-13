import os
import cv2
import numpy as np
import imageio
from tqdm import tqdm

from simulation import (
    SimulationConfig,
    OpticalRenderer,
    BackgroundEngine,
    FluidBedKinematics
)

def create_preview_animation(
    num_frames: int = 30,
    output_gif: str = "outputs/demo/fluid_bed_simulation.gif",
    output_mp4: str = "outputs/demo/fluid_bed_simulation.mp4",
    fps: int = 12,
    downscale: float = 0.5
):
    """
    Renders a continuous physical sequence of freely moving and tumbling pellets,
    producing an animated GIF and MP4 with annotated tracking boxes.
    """
    config = SimulationConfig(
        num_particles_active=30,
        prob_single=0.40,
        prob_dimer=0.30,
        prob_trimer=0.18,
        prob_cluster=0.12
    )
    
    bg_engine = BackgroundEngine(config)
    bg_plate = bg_engine.extract_from_real_data("Real-Data/T11/B4")
    
    renderer = OpticalRenderer(config)
    kinematics = FluidBedKinematics(config)
    
    colors = {
        0: (0, 230, 0),       # Green: Single
        1: (0, 165, 255),     # Orange: Dimer (2)
        2: (0, 0, 255),       # Red: Trimer (3)
        3: (255, 0, 255)      # Magenta: Cluster (4+)
    }
    
    gif_frames = []
    
    # Target dimensions for output video/gif
    out_w = int(config.image_width * downscale)
    out_h = int(config.image_height * downscale)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_gif)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_mp4, fourcc, float(fps), (out_w, out_h), True)
    
    print(f"Rendering {num_frames} frames animation...")
    for frame_idx in tqdm(range(num_frames)):
        kinematics.step(dt=1.0)
        bg = bg_engine.get_background(jitter_brightness=True)
        img, annotations = renderer.render_scene(bg, kinematics.entities)
        
        # Draw annotations
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for a in annotations:
            cid = a['class_id']
            col = colors.get(cid, (255, 255, 0))
            x1, y1, x2, y2 = a['bbox_xyxy']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(
                annotated, f"ID:{a['id']} {a['class_name']}",
                (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA
            )
            if 'polygon' in a and len(a['polygon']) >= 3:
                pts = np.array(a['polygon'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=col, thickness=1)
                
        # Resize for smooth viewing / GIF export
        resized = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)
        video_writer.write(resized)
        
        # Convert BGR to RGB for imageio GIF
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        gif_frames.append(rgb_frame)
        
    video_writer.release()
    print(f"Saved video to {output_mp4}")
    
    # Save GIF
    imageio.mimsave(output_gif, gif_frames, fps=fps, loop=0)
    print(f"Saved animated GIF to {output_gif}")

if __name__ == '__main__':
    create_preview_animation(num_frames=24, fps=10, downscale=0.4)

