from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from video_processor import VideoProcessor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

processor = VideoProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "Padel tracker API is running"})

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload and process video"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: mp4, mov, avi, mkv"}), 400
    
    try:
        # Save uploaded file
        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.{ext}")
        file.save(input_path)
        
        return jsonify({
            "job_id": job_id,
            "status": "uploaded",
            "message": "Video uploaded successfully. Use /process endpoint to start processing."
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process/<job_id>', methods=['POST'])
def process_video(job_id):
    """Process the uploaded video"""
    try:
        # Find the uploaded file
        input_file = None
        for ext in ['mp4', 'mov', 'avi', 'mkv']:
            path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.{ext}")
            if os.path.exists(path):
                input_file = path
                break
        
        if not input_file:
            return jsonify({"error": "Video not found. Please upload first."}), 404
        
        # Process video
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}.mp4")
        stats = processor.process_video(input_file, output_path)
        
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "statistics": stats,
            "output_video": f"/download/{job_id}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

@app.route('/download/<job_id>', methods=['GET'])
def download_video(job_id):
    """Download processed video"""
    try:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}.mp4")
        
        if not os.path.exists(output_path):
            return jsonify({"error": "Processed video not found"}), 404
        
        return send_file(output_path, mimetype='video/mp4', as_attachment=True, 
                        download_name=f"processed_{job_id}.mp4")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats/<job_id>', methods=['GET'])
def get_stats(job_id):
    """Get statistics for a processed video"""
    # For InstantDB integration later
    return jsonify({"message": "Stats endpoint - integrate with InstantDB here"})

if __name__ == '__main__':
    print("Starting Padel Tracker API...")
    print("Upload endpoint: POST /upload")
    print("Process endpoint: POST /process/<job_id>")
    print("Download endpoint: GET /download/<job_id>")
    app.run(host='0.0.0.0', port=5000, debug=True)
