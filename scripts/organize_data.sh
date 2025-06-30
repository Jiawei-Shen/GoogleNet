#!/bin/bash

# --- Script Start ---

# Check if a directory path is provided as an argument.
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_dataset_directory>"
    echo "Please provide the path to the directory containing 'true' and 'false' subfolders."
    exit 1
fi

# The main directory containing the 'true' and 'false' folders, passed as an argument.
BASE_DIR="$1"

# Navigate into the target directory. Exit if it fails.
cd "$BASE_DIR" || { echo "Error: Directory '$BASE_DIR' not found."; exit 1; }

echo "Successfully changed to directory: $(pwd)"

# Create the new directory structure for train, validation, and test sets.
echo "Creating new directory structure: train, val, test"
mkdir -p train/true train/false val/true val/false test/true test/false

# --- Process 'true' files ---
echo "--- Processing 'true' files ---"

# Check if the 'true' directory has files to move.
if [ -d "true" ] && [ "$(ls -A true)" ]; then
    # Get the total count of files in the 'true' directory.
    TRUE_COUNT=$(ls -1q true/ | wc -l)
    echo "Found $TRUE_COUNT files in 'true' directory."

    # Calculate split sizes (70% train, 15% val, 15% test).
    TRAIN_TRUE_COUNT=$((TRUE_COUNT * 70 / 100))
    VAL_TRUE_COUNT=$((TRUE_COUNT * 15 / 100))

    echo "Splitting 'true' files: $TRAIN_TRUE_COUNT for train, $VAL_TRUE_COUNT for validation."

    # Move 70% of 'true' files to train/true.
    # We list files, shuffle them, take the top N, and move them.
    ls -1q true/ | shuf | head -n $TRAIN_TRUE_COUNT | xargs -I {} mv "true/{}" "train/true/"

    # Move 15% of the original count to val/true from the remaining files.
    ls -1q true/ | shuf | head -n $VAL_TRUE_COUNT | xargs -I {} mv "true/{}" "val/true/"

    # Move all remaining 'true' files to test/true.
    echo "Moving remaining 'true' files to test/true."
    mv true/* test/true/
else
    echo "Warning: 'true' directory is empty or does not exist. Skipping."
fi


# --- Process 'false' files ---
echo "--- Processing 'false' files ---"

# Check if the 'false' directory has files to move.
if [ -d "false" ] && [ "$(ls -A false)" ]; then
    # Get the total count of files in the 'false' directory.
    FALSE_COUNT=$(ls -1q false/ | wc -l)
    echo "Found $FALSE_COUNT files in 'false' directory."

    # Calculate split sizes (70% train, 15% val, 15% test).
    TRAIN_FALSE_COUNT=$((FALSE_COUNT * 70 / 100))
    VAL_FALSE_COUNT=$((FALSE_COUNT * 15 / 100))

    echo "Splitting 'false' files: $TRAIN_FALSE_COUNT for train, $VAL_FALSE_COUNT for validation."

    # Move 70% of 'false' files to train/false.
    ls -1q false/ | shuf | head -n $TRAIN_FALSE_COUNT | xargs -I {} mv "false/{}" "train/false/"

    # Move 15% of the original count to val/false from the remaining files.
    ls -1q false/ | shuf | head -n $VAL_FALSE_COUNT | xargs -I {} mv "false/{}" "val/false/"

    # Move all remaining 'false' files to test/false.
    echo "Moving remaining 'false' files to test/false."
    mv false/* test/false/
else
    echo "Warning: 'false' directory is empty or does not exist. Skipping."
fi


# --- Clean up ---
echo "--- Cleaning up ---"

# Remove the now-empty original directories.
rmdir true false 2>/dev/null || echo "Original 'true'/'false' directories already removed or were not found."

echo "Splitting complete."
echo "Final directory structure:"
ls -R
