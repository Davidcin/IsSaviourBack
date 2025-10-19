#!/bin/bash

# API Test Script

# Run this after starting Flask server to test all endpoints


BASE_URL="http://localhost:5000"
TEST_VIDEO="tennis-analyzer-YOLOv8/inputs/videos/input_video.mp4"


echo "================================================"
echo "Padel Tracker API Test Script"
echo "================================================"
echo ""

# Test 1: Health Check
echo "TEST 1: Health Check"
echo "---------------------"
curl -s $BASE_URL/health | jq .
echo ""
echo ""

# Test 2: Upload Video
echo "TEST 2: Upload Video"
echo "---------------------"
if [ ! -f "$TEST_VIDEO" ]; then
    echo "Error: Test video not found at $TEST_VIDEO"
    exit 1
fi


UPLOAD_RESPONSE=$(curl -s -X POST -F "video=@$TEST_VIDEO" $BASE_URL/upload)
echo $UPLOAD_RESPONSE | jq .
echo ""

# Extract job_id from response
JOB_ID=$(echo $UPLOAD_RESPONSE | jq -r '.job_id')

if [ "$JOB_ID" == "null" ] || [ -z "$JOB_ID" ]; then
    echo "Error: Failed to get job_id from upload"
    exit 1
fi



echo "Job ID: $JOB_ID"
echo ""
echo ""
# Test 3: Process Video
echo "TEST 3: Process Video"
echo "---------------------"
echo "This will take 30-60 seconds..."
echo ""


PROCESS_RESPONSE=$(curl -s -X POST $BASE_URL/process/$JOB_ID)
echo $PROCESS_RESPONSE | jq .
echo ""
echo ""



# Test 4: Download Info
echo "TEST 4: Download Link"
echo "---------------------"
echo "Download URL: $BASE_URL/download/$JOB_ID"
echo ""
echo "To download, run:"
echo "curl -O $BASE_URL/download/$JOB_ID"
echo ""
echo ""


echo "================================================"
echo "All tests completed!"
echo "================================================"
echo ""
echo "Job ID for reference: $JOB_ID"
echo ""