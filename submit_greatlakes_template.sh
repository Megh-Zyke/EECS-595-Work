#!/bin/bash
#SBATCH --job-name=generate_personas
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --output=generate_personas_%j.out


MODEL="Qwen/Qwen2.5-7B-Instruct"

# Input CSV and outputs
INPUT_FILE="job_descriptions.csv"
OUT_DIR="personas_output"

# Number of rows processed per chunk
# For 474K rows, use larger chunks to reduce overhead (every chunk reloads model)
CHUNK_SIZE=500

# Device: 'cuda' for GPU, 'cpu' for CPU-only
DEVICE="cuda"

# Optional: Process only a subset of rows (useful for parallel jobs)
# Set via SLURM --export flag or change here
START_ROW="${START:-0}"
END_ROW="${END:-5251}"

## ---- RUN ----
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "========================================================================"
echo "Job started on $(hostname) at $(date)"
echo "========================================================================"
echo "Input: $INPUT_FILE"
echo "Model: $MODEL"
echo "Chunk size: $CHUNK_SIZE"
echo "Device: $DEVICE"
echo "Processing rows: $START_ROW to $END_ROW"
echo "========================================================================"

# Create unique output directory for this job (to allow parallel jobs)
if [ "$START_ROW" -eq 0 ] && [ "$END_ROW" -eq 474026 ]; then
    # Single job processing all rows
    OUT_DIR="personas_output"
else
    # Parallel job - use unique directory
    OUT_DIR="personas_output_${SLURM_JOB_ID}_${START_ROW}_${END_ROW}"
fi

mkdir -p "$OUT_DIR"
mkdir -p logs

# Count rows to process
TOTAL_TO_PROCESS=$((END_ROW - START_ROW))
echo "Total rows to process: $TOTAL_TO_PROCESS"

# Process rows in chunks
# First chunk without --append, subsequent chunks with --append to accumulate results
CHUNK_NUM=0
CURRENT_START=$START_ROW

echo ""
echo "Starting processing..."
echo ""

while [ $CURRENT_START -lt $END_ROW ]; do
  CHUNK_NUM=$((CHUNK_NUM + 1))
  CURRENT_END=$((CURRENT_START + CHUNK_SIZE))
  if [ $CURRENT_END -gt $END_ROW ]; then
    CURRENT_END=$END_ROW
  fi

  ROWS_IN_CHUNK=$((CURRENT_END - CURRENT_START))
  echo "[$CHUNK_NUM] Processing rows $CURRENT_START .. $((CURRENT_END-1)) ($ROWS_IN_CHUNK rows)"

  # Run Python script
  if [ $CURRENT_START -eq $START_ROW ]; then
    # First chunk: overwrite any existing output files
    python generate_personas.py \
      --input "$INPUT_FILE" \
      --outdir "$OUT_DIR" \
      --model "$MODEL" \
      --device "$DEVICE" \
      --start $CURRENT_START --end $CURRENT_END \
      2>&1 | tee -a "logs/chunk_$CHUNK_NUM.log"
  else
    # Subsequent chunks: append to existing output
    python generate_personas.py \
      --input "$INPUT_FILE" \
      --outdir "$OUT_DIR" \
      --model "$MODEL" \
      --device "$DEVICE" \
      --start $CURRENT_START --end $CURRENT_END \
      --append \
      2>&1 | tee -a "logs/chunk_$CHUNK_NUM.log"
  fi

  if [ $? -ne 0 ]; then
    echo "ERROR: Chunk $CHUNK_NUM failed! Check logs/chunk_$CHUNK_NUM.log"
    exit 1
  fi

  CURRENT_START=$CURRENT_END
done

echo ""
echo "========================================================================"
echo "All $CHUNK_NUM chunks processed successfully!"
echo "========================================================================"
echo "Output files:"
echo "  JSONL: $OUT_DIR/personas.jsonl (one persona per line)"
echo "  CSV:   $OUT_DIR/annotated_jobs_with_persona.csv (all data + persona_json column)"
echo "Logs:   logs/chunk_*.log"
echo "========================================================================"
echo "Job finished at $(date)"
