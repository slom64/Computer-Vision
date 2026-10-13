import os
import argparse
import random
import cv2
import numpy as np
from tqdm import tqdm

from simulation import (
    SimulationConfig,
    OpticalRenderer,
    BackgroundEngine,
    FluidBedKinematics,
    DatasetExporter
)
from simulation.hardware_profiler import profile_hardware, print_hardware_summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate photorealistic synthetic fluid-bed pellet dataset with multi-angle lighting and macro-agglomerates."
    )
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to generate (auto-selected by GPU if not specified).")
    parser.add_argument("--real_data_dir", type=str, default="Real-Data/T11/B4", help="Path to real dataset folder for background extraction.")
    parser.add_argument("--output_dir", type=str, default="outputs/synthetic_dataset", help="Output directory.")
    parser.add_argument("--mode", type=str, default="binary", choices=["detailed", "binary"], help="Classification mode.")
    parser.add_argument("--particles_per_frame", type=int, default=240, help="Number of concurrent particles in fluid bed chamber.")
    parser.add_argument("--max_cluster_size", type=int, default=20, help="Maximum pellets connected in macro-agglomerates (e.g. 10 to 20).")
    parser.add_argument("--lighting_mode", type=str, default="dynamic", choices=["dynamic", "coaxial", "side_grazing", "high_contrast"], help="Photometric illumination mode.")
    parser.add_argument("--prob_single", type=float, default=0.30, help="Probability of single pellet.")
    parser.add_argument("--prob_dimer", type=float, default=0.22, help="Probability of 2-pellet agglomerate.")
    parser.add_argument("--prob_trimer", type=float, default=0.16, help="Probability of 3-pellet agglomerate.")
    parser.add_argument("--prob_cluster", type=float, default=0.18, help="Probability of 4-6 cluster.")
    parser.add_argument("--prob_macro", type=float, default=0.14, help="Probability of 7-20 macro-agglomerate.")
    parser.add_argument("--split_ratio", type=float, default=0.85, help="Train/val split ratio.")
    parser.add_argument("--export_seg", action="store_true", default=True, help="Export YOLO segmentation polygons.")
    parser.add_argument("--export_labelme", action="store_true", default=True, help="Export LabelMe JSON files.")
    parser.add_argument("--num_previews", type=int, default=8, help="Number of visually annotated preview images to save.")
    parser.add_argument("--sequence_mode", action="store_true", default=True, help="Simulate continuous physical movement over time.")
    parser.add_argument("--save_video", action="store_true", default=True, help="Compile rendered sequence into an MP4 video.")
    parser.add_argument("--auto_gpu", action="store_true", default=True, help="Auto-detect GPU and configure optimal volume.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def draw_visual_annotations(img: np.ndarray, annotations: list, config: SimulationConfig) -> np.ndarray:
    annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = {
        0: (0, 230, 0),       # Green: Single
        1: (0, 165, 255),     # Orange: Dimer (2)
        2: (0, 0, 255),       # Red: Trimer (3)
        3: (255, 0, 255),     # Magenta: Cluster (4-6)
        4: (255, 255, 0)      # Cyan: Macro-Agglomerate (7-20)
    }
    
    for annot in annotations:
        cid = annot['class_id']
        c = colors.get(cid, (255, 255, 0))
        x1, y1, x2, y2 = annot['bbox_xyxy']
        cv2.rectangle(annotated, (x1, y1), (x2, y2), c, 2)
        
        morph_info = f" [{annot.get('morphology', '')}]" if annot.get('morphology') else ""
        label_str = f"{annot['class_name']} (N={annot['count']}){morph_info}"
        cv2.putText(
            annotated, label_str, (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA
        )
        
        if 'polygon' in annot and len(annot['polygon']) >= 3:
            pts = np.array(annot['polygon'], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=c, thickness=1)
            
    return annotated

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 1. Hardware auto-profiling
    hw = profile_hardware()
    print_hardware_summary(hw)
    
    num_frames = args.num_frames
    if num_frames is None:
        num_frames = hw.recommended_dataset_frames
        print(f"Auto-selected dataset volume based on {hw.device_name}: {num_frames} frames.")
        
    config = SimulationConfig(
        classification_mode=args.mode,
        num_particles_active=args.particles_per_frame,
        max_cluster_size=args.max_cluster_size,
        lighting_mode=args.lighting_mode,
        randomize_lighting=True,
        prob_single=args.prob_single,
        prob_dimer=args.prob_dimer,
        prob_trimer=args.prob_trimer,
        prob_cluster=args.prob_cluster,
        prob_macro=args.prob_macro
    )
    
    print("\n" + "=" * 70)
    print(" [SYNTHETIC DATASET GENERATION]")
    print("=" * 70)
    print(f" Target Frames        : {num_frames}")
    print(f" Active Pellets/Frame : {config.num_particles_active}")
    print(f" Max Agglomerate Size : Up to {config.max_cluster_size} connected pellets")
    print(f" Lighting Mode        : {config.lighting_mode.upper()} (Multi-angle & drift enabled)")
    print(f" Classification Mode  : {args.mode}")
    print(f" Class Breakdown Map  : {config.get_class_names()}")
    print(f" Background Source    : {args.real_data_dir}")
    print(f" Output Target        : {args.output_dir}")
    print("=" * 70)
    
    # Initialize engines
    bg_engine = BackgroundEngine(config)
    bg_plate = bg_engine.extract_from_real_data(real_data_dir=args.real_data_dir)
    print(f"Background chamber plate initialized ({bg_plate.shape[1]}x{bg_plate.shape[0]}).")
    
    renderer = OpticalRenderer(config)
    kinematics = FluidBedKinematics(config)
    exporter = DatasetExporter(config, output_dir=args.output_dir)
    exporter.setup_directories()
    exporter.open_tracking_logger()
    
    previews_dir = os.path.join(args.output_dir, 'previews')
    os.makedirs(previews_dir, exist_ok=True)
    
    video_writer = None
    if args.save_video:
        video_path = os.path.join(args.output_dir, 'simulated_fluid_bed.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (config.image_width, config.image_height), False)
        print(f"Compiling simulation video to: {video_path}")
        
    num_train = int(num_frames * args.split_ratio)
    total_annotations_count = 0
    class_counts = {cid: 0 for cid in config.get_class_names().keys()}
    max_pellets_in_single_agglom = 0
    
    print(f"\nRendering {num_frames} high-resolution frames with dynamic lighting...")
    for frame_idx in tqdm(range(num_frames)):
        split = 'train' if frame_idx < num_train else 'val'
        
        # Advance fluid-bed physics
        if args.sequence_mode:
            kinematics.step(dt=1.0)
        else:
            kinematics.initialize_population()
            
        bg = bg_engine.get_background(jitter_brightness=True)
        frame_img, annotations = renderer.render_scene(bg, kinematics.entities)
        
        # Export frame labels (YOLO bbox + YOLO segmentation + LabelMe)
        exporter.save_frame(
            frame_idx=frame_idx,
            image=frame_img,
            annotations=annotations,
            split=split,
            export_labelme=args.export_labelme,
            export_yolo_seg=args.export_seg
        )
        
        if video_writer:
            video_writer.write(frame_img)
            
        if frame_idx < args.num_previews:
            annotated_preview = draw_visual_annotations(frame_img, annotations, config)
            preview_path = os.path.join(previews_dir, f"preview_{frame_idx:04d}.jpg")
            cv2.imwrite(preview_path, annotated_preview)
            
        total_annotations_count += len(annotations)
        for a in annotations:
            class_counts[a['class_id']] += 1
            if a['count'] > max_pellets_in_single_agglom:
                max_pellets_in_single_agglom = a['count']

    exporter.close_tracking_logger()
    if video_writer:
        video_writer.release()
        
    print("\n" + "=" * 70)
    print(" [GENERATION SUMMARY]")
    print("=" * 70)
    print(f" Total Labeled Pellets/Agglomerates : {total_annotations_count:,}")
    print(f" Max Single Agglomerate Observed    : {max_pellets_in_single_agglom} pellets connected")
    print(f" Train Split / Val Split            : {num_train} train / {num_frames - num_train} val frames")
    print(" Class Distribution:")
    for cid, name in config.get_class_names().items():
        print(f"  * Class {cid} ({name}): {class_counts[cid]:,} instances")
    print(f" Dataset Path: {os.path.abspath(args.output_dir)}")
    print(f" YAML Config : {os.path.join(args.output_dir, 'dataset.yaml')}")
    print("=" * 70)

if __name__ == '__main__':
    main()
