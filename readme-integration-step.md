## Step 1: Expose Your API with ngrok (5 minutes)

**First, you need to make your local Flask API accessible from the internet:**

### Install ngrok (if you haven't):

```bash
brew install ngrok
```

### Sign up and authenticate:

1. Go to [](https://ngrok.com)<https://ngrok.com> and create free account
2. Get your auth token from dashboard
3. Run: `ngrok config add-authtoken YOUR_TOKEN`

### Start ngrok:

```bash
# In a NEW terminal (keep Flask running in another)
ngrok http 5000
```

You'll see something like:

```javascript
Forwarding   https://abc123.ngrok-free.app -> http://localhost:5000
```

**Copy that `https://abc123.ngrok-free.app` URL!** This is your public API endpoint.

---

## Step 2: Add Code to Lovable Frontend

In your Lovable project, you'll add JavaScript to call your API. Here's exactly what to add:

### A. Add the API URL constant:

```javascript
// Put this at the top of your script
const API_URL = "https://YOUR-NGROK-URL.ngrok-free.app";
```

Replace `YOUR-NGROK-URL` with the actual ngrok URL you copied.

### B. Add the upload and process functions:

```javascript
// Function to handle video upload
async function analyzeVideo(videoFile) {
    try {
        // Show loading state
        updateStatus("Uploading video...", "info");

        // Step 1: Upload
        const formData = new FormData();
        formData.append("video", videoFile);

        const uploadResponse = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!uploadResponse.ok) {
            throw new Error("Upload failed");
        }

        const uploadData = await uploadResponse.json();
        const jobId = uploadData.job_id;

        // Step 2: Process
        updateStatus("Processing video with AI...", "info");

        const processResponse = await fetch(`${API_URL}/process/${jobId}`, {
            method: "POST",
        });

        if (!processResponse.ok) {
            throw new Error("Processing failed");
        }

        const result = await processResponse.json();

        // Step 3: Display results
        displayResults(result);

        // Step 4: Provide download link
        const downloadUrl = `${API_URL}/download/${jobId}`;
        showDownloadLink(downloadUrl);

        updateStatus("Analysis complete!", "success");
    } catch (error) {
        console.error("Error:", error);
        updateStatus("Error: " + error.message, "error");
    }
}

// Function to display statistics
function displayResults(result) {
    const stats = result.statistics;

    document.getElementById("total-players").textContent =
        stats.players_detected;
    document.getElementById("avg-players").textContent =
        stats.average_players_per_frame.toFixed(1);
    document.getElementById("ball-detections").textContent =
        stats.ball_detections;
    document.getElementById("ball-hits").textContent = stats.ball_hit_count;
    document.getElementById("rally-duration").textContent =
        stats.rally_duration_seconds + "s";

    // Show stats section
    document.getElementById("stats-container").style.display = "block";
}

// Helper functions
function updateStatus(message, type) {
    const statusEl = document.getElementById("status-message");
    statusEl.textContent = message;
    statusEl.className = `status-${type}`; // status-info, status-success, status-error
}

function showDownloadLink(url) {
    const downloadLink = document.getElementById("download-link");
    downloadLink.href = url;
    downloadLink.style.display = "inline-block";
    downloadLink.textContent = "Download Processed Video";
}
```

### C. Add the HTML elements Lovable needs:

```html
<div class="upload-section">
    <input type="file" id="video-input" accept="video/*" />
    <button onclick="handleUpload()">Analyze Video</button>
</div>

<div id="status-message" class="status-info"></div>

<div id="stats-container" style="display: none;">
    <h2>Analysis Results</h2>
    <div class="stat">
        <span class="stat-label">Total Players Detected:</span>
        <span id="total-players" class="stat-value">-</span>
    </div>
    <div class="stat">
        <span class="stat-label">Average Players per Frame:</span>
        <span id="avg-players" class="stat-value">-</span>
    </div>
    <div class="stat">
        <span class="stat-label">Ball Detections:</span>
        <span id="ball-detections" class="stat-value">-</span>
    </div>
    <div class="stat">
        <span class="stat-label">Ball Hits:</span>
        <span id="ball-hits" class="stat-value">-</span>
    </div>
    <div class="stat">
        <span class="stat-label">Rally Duration:</span>
        <span id="rally-duration" class="stat-value">-</span>
    </div>

    <a id="download-link" style="display: none;">Download Processed Video</a>
</div>

<script>
    function handleUpload() {
        const input = document.getElementById("video-input");
        const file = input.files[0];

        if (!file) {
            alert("Please select a video file");
            return;
        }

        // Call the analyze function
        analyzeVideo(file);
    }
</script>
```

---

## Step 3: Test It Works

1. **Make sure Flask is running**: `python3 app.py`
2. **Make sure ngrok is running**: `ngrok http 5000`
3. **Open your Lovable site**
4. **Upload a short test video** (10-20 seconds)
5. **Watch the status updates**
6. **See the statistics appear**
7. **Download the processed video**

---

##
