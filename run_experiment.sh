#!/bin/bash

# Default values
USERNAME="anonymous"
DATASET=""
TARGET_COL="Predicted Disease"
TASK_TYPE="classification"
NUM_CLIENTS=5
NUM_ROUNDS=10

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --username) USERNAME="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --target) TARGET_COL="$2"; shift ;;
        --task) TASK_TYPE="$2"; shift ;;
        --clients) NUM_CLIENTS="$2"; shift ;;
        --rounds) NUM_ROUNDS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required (e.g., --dataset your_data.csv)"
    exit 1
fi

DATA_PATH="data/$DATASET"
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH. Did you put it in the data/ folder?"
    exit 1
fi

echo "=================================================="
echo "Starting FL Simulation for user: $USERNAME"
echo "Dataset: $DATA_PATH | Target: $TARGET_COL"
echo "=================================================="

# Use virtual environment python if available
PYTHON_CMD="python"
if [ -f ".fed/bin/python" ]; then
    PYTHON_CMD=".fed/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_CMD=".venv/Scripts/python.exe"
fi

$PYTHON_CMD fl_simulation.py \
    --data_path "$DATA_PATH" \
    --target_col "$TARGET_COL" \
    --username "$USERNAME" \
    --task_type "$TASK_TYPE" \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS

if [ $? -eq 0 ]; then
    echo "Simulation complete! Generating reports..."
    # Robustly find the exact JSON file just created for this user
    LATEST_JSON=$(ls -t results/simulation_results_${USERNAME}_*.json | head -1)
    
    $PYTHON_CMD generate_report.py --result_file "$LATEST_JSON"
    echo "=================================================="
    echo "All Done! Check the reports/ folder."
else
    echo "Simulation failed. See above errors."
fi
