## Command to run
```
CTX_SIZE="32768" MODEL_PATH="/hpc/hdong/models/Qwen3.5-27B-Q3_K_S.gguf" ./setup_and_run.sh
```

## Commands for SLURM

### Launch job
```
sbatch run_chatbot_slurm.sh
```

### view job
```
squeue
```

### cancel job
```
scancel -u <username>
```