"""
Quick test script to verify the video processor works
"""
from video_processor import VideoProcessor
import os

def test_processor():
    print("Testing Video Processor...")
    
    # Use one of the sample videos from the tennis analyzer
    test_video = "tennis-analyzer-YOLOv8/inputs/videos/input_video.mp4"
    if not os.path.exists(test_video):
        print(f"Error: Test video not found at {test_video}")
        print("Available videos:")
        video_dir = "tennis-analyzer-YOLOv8/inputs/videos"
        if os.path.exists(video_dir):
            for f in os.listdir(video_dir):
                print(f" - {f}")
            return

    # Create processor
    processor = VideoProcessor()



    # Process just first 60 frames for quick test

    output = "test_output.mp4"
    print(f"\nProcessing {test_video} (first 60 frames)...")
    stats = processor.process_video(test_video, output, max_frames=60)

    print("\n=== RESULTS ===")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Players detected: {stats['players_detected']}")
    print(f"Ball detections: {stats['ball_detections']}")
    print(f"Ball hits estimated: {stats['ball_hit_count']}")
    print(f"Average players per frame: {stats['average_players_per_frame']:.2f}")
    print(f"Rally duration: {stats['rally_duration_seconds']} seconds")
    print(f"\nOutput video saved to: {output}")
    print("\n✓ Test completed successfully!")



if __name__ == "__main__":
    test_processor()