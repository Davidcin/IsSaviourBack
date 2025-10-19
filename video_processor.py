import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

class VideoProcessor:
    def __init__(self):
        """Initialize YOLOv8 model"""
        print("Loading YOLOv8 model...")
        # Using medium YOLOv8 model for better small object detection
        self.model = YOLO('yolov8m.pt')  # medium model
        print("Model loaded successfully!")
    
    def process_video(self, input_path, output_path, max_frames=None):
        """
        Process video and generate statistics
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            max_frames: Limit processing (useful for testing)
        
        Returns:
            Dictionary with statistics
        """
        print(f"Processing video: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Statistics tracking
        stats = {
            "total_frames": 0,
            "players_detected": 0,
            "ball_detections": 0,
            "player_positions": [],
            "ball_positions": [],
            "average_players_per_frame": 0,
            "ball_hit_count": 0,
            "rally_duration_seconds": 0
        }
        
        frame_count = 0
        player_counts = []
        ball_detected_frames = []
        
        # Track ball movement for hit detection
        prev_ball_pos = None
        ball_velocity_threshold = 50  # pixels/frame for detecting hits
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Limit frames for faster processing during hackathon
            if max_frames and frame_count > max_frames:
                break
            
            # Process every 2nd frame for speed (adjust as needed)
            if frame_count % 2 != 0:
                out.write(frame)
                continue
            
            try:
                # Run YOLOv8 detection
                results = self.model(frame, verbose=False)
                
                # Process detections
                people_count = 0
                ball_detected = False
                current_ball_pos = None
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Class 0 is 'person' in COCO dataset
                        if cls == 0 and conf > 0.5:
                            people_count += 1
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            stats["player_positions"].append((center_x, center_y))
                            
                            # Draw bounding box for person
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                        (0, 255, 0), 2)
                            cv2.putText(frame, f"Player {people_count}", (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Class 32 is 'sports ball' in COCO dataset
                        elif cls == 32 and conf > 0.1:
                            ball_detected = True
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            current_ball_pos = (center_x, center_y)
                            stats["ball_positions"].append((center_x, center_y))
                            
                            # Draw ball
                            cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)
                            cv2.putText(frame, "Ball", (center_x-20, center_y-20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # Detect ball hits (rapid velocity change)
                            if prev_ball_pos:
                                dx = current_ball_pos[0] - prev_ball_pos[0]
                                dy = current_ball_pos[1] - prev_ball_pos[1]
                                velocity = np.sqrt(dx**2 + dy**2)
                                
                                if velocity > ball_velocity_threshold:
                                    stats["ball_hit_count"] += 1
                
                # Update stats
                player_counts.append(people_count)
                if ball_detected:
                    ball_detected_frames.append(frame_count)
                    stats["ball_detections"] += 1
                
                prev_ball_pos = current_ball_pos
                
                # Add frame info
                info_text = f"Frame: {frame_count} | Players: {people_count} | Ball Hits: {stats['ball_hit_count']}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                # Write processed frame
                out.write(frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames...")
            
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                out.write(frame)
                continue
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate final statistics
        stats["total_frames"] = frame_count
        stats["players_detected"] = sum(player_counts)
        stats["average_players_per_frame"] = np.mean(player_counts) if player_counts else 0
        
        # Estimate rally duration (time when ball was visible)
        if ball_detected_frames:
            rally_frames = len(ball_detected_frames)
            stats["rally_duration_seconds"] = round(rally_frames / fps, 2)
        
        print(f"Processing complete! Output saved to: {output_path}")
        print(f"Statistics: {stats}")
        
        return stats
