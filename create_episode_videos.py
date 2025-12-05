#!/usr/bin/env python3
"""
Script to create videos from successful episode runs in EmbodiedBench.
- Finds successful episodes in eb_alfred and eb_habitat
- Sorts by score and selects top N
- Creates videos with action legends at 1 FPS
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from collections import defaultdict


def find_successful_episodes(base_path: Path, benchmark_name: str) -> List[Dict]:
    """
    Find all successful episodes in a benchmark directory.
    
    Args:
        base_path: Path to the benchmark directory (e.g., running/eb_alfred)
        benchmark_name: Name of the benchmark (e.g., 'eb_alfred')
    
    Returns:
        List of dicts containing episode info
    """
    successful_episodes = []
    
    # Iterate through all experiment runs
    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Filter: only process runs starting with "Qwen2.5-VL-7B-Instruct"
        if not run_dir.name.startswith("Qwen2.5-VL-7B-Instruct"):
            continue
            
        # Iterate through subdirectories (base, common_sense, etc.)
        for subdir in run_dir.iterdir():
            if not subdir.is_dir() or subdir.name in ['images', 'results', 'locks', 'signals', 'dones']:
                continue
            
            # Find all episode JSON files
            for json_file in subdir.glob("episode_*_step_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        # Read all lines (each line is a JSON object representing a step)
                        lines = f.readlines()
                        if not lines:
                            continue
                        
                        # Parse the last line to get final state
                        final_state = json.loads(lines[-1])
                        
                        # Check if task was successful
                        if final_state.get('task_success', 0.0) == 1.0:
                            # Extract episode info
                            episode_name = json_file.stem  # e.g., episode_1_step_12
                            episode_num = episode_name.split('_')[1]
                            
                            # Get the score (using task_progress as the metric)
                            score = final_state.get('task_progress', 0.0)
                            
                            # Parse all steps to get actions
                            steps_data = []
                            for line in lines:
                                step_data = json.loads(line)
                                steps_data.append({
                                    'env_step': step_data.get('env_step', 0),
                                    'action_description': step_data.get('action_description', 'Unknown action'),
                                    'action_id': step_data.get('action_id', -1),
                                    'last_action_success': step_data.get('last_action_success', 0.0)
                                })
                            
                            # Check if images directory exists
                            images_dir = subdir / 'images' / f'episode_{episode_num}'
                            if images_dir.exists():
                                successful_episodes.append({
                                    'benchmark': benchmark_name,
                                    'run_name': run_dir.name,
                                    'subdir_name': subdir.name,
                                    'episode_num': episode_num,
                                    'score': score,
                                    'instruction': final_state.get('instruction', 'No instruction'),
                                    'images_dir': images_dir,
                                    'steps_data': steps_data,
                                    'json_file': json_file
                                })
                
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
    
    return successful_episodes


def sort_and_filter_episodes(episodes: List[Dict], top_n: int = 10) -> List[Dict]:
    """
    Sort episodes by score and return top N.
    
    Args:
        episodes: List of episode dicts
        top_n: Number of top episodes to return
    
    Returns:
        Sorted list of top N episodes
    """
    # Sort by score (descending)
    sorted_episodes = sorted(episodes, key=lambda x: x['score'], reverse=True)
    return sorted_episodes[:top_n]


def add_text_to_frame(frame: np.ndarray, text: str, step_num: int, instruction: str = None) -> np.ndarray:
    """
    Add text overlay to frame at top-left.
    
    Args:
        frame: Image frame as numpy array
        text: Text to display
        step_num: Step number
        instruction: Task instruction (shown only on first frame)
    
    Returns:
        Frame with text overlay
    """
    # Create a copy to avoid modifying original
    frame_with_text = frame.copy()
    
    # Prepare text
    lines = []
    
    # Add task instruction on first frame
    if instruction and step_num == 0:
        lines.append(f"Task: {instruction}")
        lines.append("")  # Empty line for spacing
    
    lines.extend([
        f"Step {step_num}",
        f"Action: {text}"
    ])
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    bg_color = (0, 0, 0)  # Black background
    padding = 10
    line_spacing = 35
    
    # Calculate text sizes and create background
    y_offset = padding
    max_width = 0
    
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_width = max(max_width, text_width)
    
    # Draw semi-transparent background rectangle
    bg_height = len(lines) * line_spacing + padding * 2
    bg_width = max_width + padding * 2
    
    # Create overlay for semi-transparency
    overlay = frame_with_text.copy()
    cv2.rectangle(overlay, (0, 0), (bg_width, bg_height), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame_with_text, 0.3, 0, frame_with_text)
    
    # Draw text
    for i, line in enumerate(lines):
        y_position = y_offset + (i + 1) * line_spacing
        cv2.putText(frame_with_text, line, (padding, y_position), 
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return frame_with_text


def create_video(episode_info: Dict, output_path: Path, fps: float = 0.5):
    """
    Create video from episode images with action legends.
    
    Args:
        episode_info: Dict containing episode information
        output_path: Path where video will be saved
        fps: Frames per second (default: 0.5)
    """
    images_dir = episode_info['images_dir']
    steps_data = episode_info['steps_data']
    
    # Get all image files sorted by step number
    image_files = sorted(images_dir.glob('episode_*_step_*.png'), 
                        key=lambda x: int(x.stem.split('_')[-1]))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Read first image to get dimensions
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"Failed to read first image: {image_files[0]}")
        return
    
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Failed to create video writer for {output_path}")
        return
    
    # Create a mapping of step numbers to actions
    step_to_action = {step['env_step']: step['action_description'] for step in steps_data}
    
    # Get the task instruction
    instruction = episode_info.get('instruction', 'No instruction')
    
    # Process each image
    for i, img_file in enumerate(image_files):
        # Extract step number from filename
        step_num = int(img_file.stem.split('_')[-1])
        
        # Read image
        frame = cv2.imread(str(img_file))
        if frame is None:
            print(f"Failed to read image: {img_file}")
            continue
        
        # Get action description for this step
        action = step_to_action.get(step_num, "No action recorded")
        
        # Add text overlay (include instruction only on first frame)
        frame_with_text = add_text_to_frame(
            frame, 
            action, 
            step_num, 
            instruction if step_num == 0 else None
        )
        
        # Check if this is the last frame
        is_last_frame = (i == len(image_files) - 1)
        
        if is_last_frame:
            # Triple the duration of the last frame by writing it 3 times
            for _ in range(3):
                video_writer.write(frame_with_text)
        else:
            # Write frame to video normally
            video_writer.write(frame_with_text)
    
    # Release video writer
    video_writer.release()
    print(f"Created video: {output_path}")


def main():
    """Main function to process episodes and create videos."""
    
    # Configuration
    TOP_N = 10
    FPS = 0.5
    
    # Paths
    running_dir = Path('running')
    videos_dir = Path('videos')
    
    # Create videos directory
    videos_dir.mkdir(exist_ok=True)
    
    # Process each benchmark
    benchmarks = {
        'eb_alfred': running_dir / 'eb_alfred',
        'eb_habitat': running_dir / 'eb_habitat'
    }
    
    for benchmark_name, benchmark_path in benchmarks.items():
        if not benchmark_path.exists():
            print(f"Benchmark directory not found: {benchmark_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {benchmark_name}...")
        print(f"{'='*60}\n")
        
        # Find successful episodes
        print(f"Finding successful episodes in {benchmark_path}...")
        successful_episodes = find_successful_episodes(benchmark_path, benchmark_name)
        print(f"Found {len(successful_episodes)} successful episodes")
        
        if not successful_episodes:
            print(f"No successful episodes found for {benchmark_name}")
            continue
        
        # Sort and filter top N
        print(f"Selecting top {TOP_N} episodes by score...")
        top_episodes = sort_and_filter_episodes(successful_episodes, TOP_N)
        
        # Create benchmark-specific output directory
        benchmark_videos_dir = videos_dir / benchmark_name
        benchmark_videos_dir.mkdir(exist_ok=True)
        
        # Create videos
        print(f"\nCreating videos...")
        for i, episode in enumerate(top_episodes, 1):
            print(f"\n[{i}/{len(top_episodes)}] Episode {episode['episode_num']} "
                  f"(Score: {episode['score']:.3f})")
            print(f"  Run: {episode['run_name']}")
            print(f"  Subdir: {episode['subdir_name']}")
            print(f"  Instruction: {episode['instruction']}")
            
            # Create video filename with shortened run name
            # Extract key parts from run name for a shorter identifier
            run_parts = episode['run_name'].split('_')
            # Take first few meaningful parts (model name, date, and key identifier)
            if len(run_parts) >= 2:
                short_run_name = f"{run_parts[0]}_{run_parts[1]}"
            else:
                short_run_name = episode['run_name'][:50]  # Truncate if needed
            
            video_filename = (f"{short_run_name}_"
                            f"rank{i:02d}_ep{episode['episode_num']}_"
                            f"score{episode['score']:.3f}_"
                            f"{episode['subdir_name']}_{FPS}fps.mp4")
            video_path = benchmark_videos_dir / video_filename
            
            # Create video
            create_video(episode, video_path, FPS)
        
        print(f"\n{'='*60}")
        print(f"Completed {benchmark_name}")
        print(f"Videos saved to: {benchmark_videos_dir}")
        print(f"{'='*60}\n")
    
    print("\n" + "="*60)
    print("All videos created successfully!")
    print(f"Videos saved to: {videos_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
