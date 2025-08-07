# ASI-Arch Breakthrough Detection Guide

This guide explains how to monitor the ASI-Arch pipeline for architectural breakthroughs and successful architecture discoveries.

## 1. Real-time Pipeline Log Monitoring

The pipeline provides real-time feedback showing key performance indicators:

### Breakthrough Indicators to Watch For:
```bash
# Success patterns in pipeline logs:
[INFO] Program evolution successful, generated program: <architecture_name>
[INFO] Training successful for <architecture_name>
[INFO] Program <architecture_name> evaluation successful

# Performance improvement signs:
Training successful for <architecture_name>
# Look for improved metrics in training results
```

### Key Log Locations:
- **Pipeline logs**: Real-time stdout during `python pipeline.py`
- **Detailed logs**: `pipeline/logs/agent_calls/`
- **Training results**: `pipeline/files/analysis/loss.csv` and `benchmark.csv`

## 2. Database API Monitoring Commands

Use these commands to check for high-performing architectures:

### Top Candidates Query:
```bash
# Get top 10 performing candidates (names and basic info)
curl -X GET "http://localhost:8001/candidates/top-k/10" | jq '.[] | {name: .name, time: .time}'
```

### Recent High-Scoring Elements:
```bash
# Check latest 5 elements with names only
curl -X GET "http://localhost:8001/elements/top-k/5" | jq '.[] | {name: .name, time: .time}'

# Get specific element with score by name
curl -X GET "http://localhost:8001/elements/with-score/by-name/ARCHITECTURE_NAME" | jq '.[] | {name: .name, score: .score, time: .time}'
```

### Database Statistics:
```bash
# Get overall database statistics
curl -X GET "http://localhost:8001/stats" | jq '.'

# Get candidate system statistics (shows highest_score)
curl -X GET "http://localhost:8001/candidates/stats" | jq '.'
```

### Architecture Details:
```bash
# Get specific architecture by name (no score)
curl -X GET "http://localhost:8001/elements/by-name/ARCHITECTURE_NAME" | jq '.[] | {name: .name, time: .time}'

# Get architecture with score (recommended)
curl -X GET "http://localhost:8001/elements/with-score/by-name/ARCHITECTURE_NAME" | jq '.[] | {name: .name, score: .score, time: .time}'

# Combined query: Get architecture name and score in one command
curl -X GET "http://localhost:8001/elements/with-score/by-name/ARCHITECTURE_NAME" | jq -r '.[] | "\(.name): \(.score)"'
```

## 3. Automated Performance Threshold Monitoring

Create a monitoring script to alert on breakthroughs:

### Simple Breakthrough Monitor:
```bash
#!/bin/bash
# Save as: monitor_breakthroughs.sh

echo "🔍 Starting ASI-Arch breakthrough monitoring..."
echo "Baseline score: 2.49 (hybrid_linear_hrm)"
echo "Monitoring for scores > 3.0..."

while true; do
    # Get best current score
    BEST_SCORE=$(curl -s "http://localhost:8001/candidates/stats" | jq -r '.data.highest_score // 0')
    BEST_NAME=$(curl -s "http://localhost:8001/candidates/top-k/1" | jq -r '.data[0].name // "unknown"')
    
    # Check for breakthrough thresholds
    if (( $(echo "$BEST_SCORE > 4.0" | bc -l) )); then
        echo "🚀 PARADIGM SHIFT: $BEST_NAME scored $BEST_SCORE at $(date)"
        # Optional: send notification, create backup, etc.
    elif (( $(echo "$BEST_SCORE > 3.5" | bc -l) )); then
        echo "🌟 MAJOR BREAKTHROUGH: $BEST_NAME scored $BEST_SCORE at $(date)"
    elif (( $(echo "$BEST_SCORE > 3.0" | bc -l) )); then
        echo "✨ SIGNIFICANT IMPROVEMENT: $BEST_NAME scored $BEST_SCORE at $(date)"
    fi
    
    # Show current best every hour
    echo "Current best: $BEST_NAME (score: $BEST_SCORE) at $(date)"
    
    sleep 300  # Check every 5 minutes
done
```

### Advanced Monitoring with Alerts:
```bash
#!/bin/bash
# Save as: advanced_monitor.sh

BREAKTHROUGH_THRESHOLD=3.0
MAJOR_BREAKTHROUGH=3.5
PARADIGM_SHIFT=4.0
LAST_BEST_SCORE=0

while true; do
    CURRENT_BEST=$(curl -s "http://localhost:8001/candidates/stats" | jq -r '.data.highest_score // 0')
    
    # Only alert on score improvements
    if (( $(echo "$CURRENT_BEST > $LAST_BEST_SCORE" | bc -l) )); then
        ARCHITECTURE=$(curl -s "http://localhost:8001/candidates/top-k/1" | jq -r '.data[0].name')
        
        if (( $(echo "$CURRENT_BEST > $PARADIGM_SHIFT" | bc -l) )); then
            echo "🚀🚀🚀 PARADIGM SHIFT DETECTED! 🚀🚀🚀"
            echo "Architecture: $ARCHITECTURE"
            echo "Score: $CURRENT_BEST (previous best: $LAST_BEST_SCORE)"
            echo "Time: $(date)"
            echo "Saving architecture details..."
            curl -s "http://localhost:8001/elements/by-name/$ARCHITECTURE" > "breakthrough_${ARCHITECTURE}_$(date +%Y%m%d_%H%M%S).json"
        elif (( $(echo "$CURRENT_BEST > $MAJOR_BREAKTHROUGH" | bc -l) )); then
            echo "🌟 MAJOR BREAKTHROUGH: $ARCHITECTURE scored $CURRENT_BEST"
        elif (( $(echo "$CURRENT_BEST > $BREAKTHROUGH_THRESHOLD" | bc -l) )); then
            echo "✨ Significant improvement: $ARCHITECTURE scored $CURRENT_BEST"
        fi
        
        LAST_BEST_SCORE=$CURRENT_BEST
    fi
    
    sleep 180  # Check every 3 minutes for improvements
done
```

## 4. File-Based Result Monitoring

Monitor pipeline output files for performance trends:

### Training Results:
```bash
# Watch training loss in real-time
tail -f pipeline/files/analysis/loss.csv

# Monitor benchmark results
tail -f pipeline/files/analysis/benchmark.csv

# Check for training errors
tail -f pipeline/files/debug/training_error.txt
```

### Generated Architectures:
```bash
# List recently generated architecture files
ls -lt pipeline/pool/*.py | head -10

# Monitor architecture generation
watch -n 30 'ls -lt pipeline/pool/*.py | head -5'
```

## 5. Performance Threshold Definitions

Based on the current baseline (hybrid_linear_hrm: 2.49):

### Breakthrough Categories:
- **Score 2.5-3.0**: Incremental improvement
- **Score 3.0-3.5**: Significant breakthrough 🔥
- **Score 3.5-4.0**: Major breakthrough 🌟  
- **Score 4.0+**: Paradigm shift 🚀

### Architecture Quality Indicators:
- **Reasoning Accuracy**: >0.80 (excellent)
- **Efficiency**: >0.90 (highly efficient)
- **Composite Score**: Combines multiple metrics
- **Training Stability**: Consistent convergence

## 6. Automated Checkpointing

### Save Breakthrough Architectures:
```bash
# Function to backup successful architectures
backup_breakthrough() {
    local architecture_name=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Create breakthrough directory
    mkdir -p breakthroughs
    
    # Save architecture details
    curl -s "http://localhost:8001/elements/by-name/$architecture_name" > \
        "breakthroughs/${architecture_name}_${timestamp}.json"
    
    # Copy source code if available
    if [ -f "pipeline/pool/${architecture_name}.py" ]; then
        cp "pipeline/pool/${architecture_name}.py" \
           "breakthroughs/${architecture_name}_${timestamp}.py"
    fi
    
    # Save current database state
    curl -s "http://localhost:8001/candidates/top-k/10" > \
        "breakthroughs/top_candidates_${timestamp}.json"
    
    echo "✅ Breakthrough architecture saved to breakthroughs/"
}
```

## 7. Integration with RAG System

The pipeline now has access to advanced research through the RAG system:

### Research Papers Available:
- **HRM (Hierarchical Reasoning Models)**: Dual-module architecture
- **Neural Attention Memory**: Persistent memory beyond context windows  
- **Logarithmic Memory Networks**: Hierarchical memory with O(log n) complexity
- **Element-wise Attention**: Linear complexity with preserved spikiness
- **MemOS**: Memory operating system for AI agents
- **Recurrent Entity Networks**: Dynamic long-term memory
- **State Space Models**: Linear complexity sequence processing

### Expected Breakthrough Patterns:
- **Memory Integration**: Architectures combining attention + persistent memory
- **Multi-timescale Processing**: System 1/System 2 reasoning implementations
- **Linear Complexity**: O(n) or better attention mechanisms with preserved performance
- **Hierarchical Reasoning**: Deep reasoning with convergence detection

## 8. Usage Examples

### Start Breakthrough Monitoring:
```bash
# Make monitoring script executable
chmod +x monitor_breakthroughs.sh

# Start monitoring in background
./monitor_breakthroughs.sh > breakthrough_log.txt 2>&1 &

# Follow the log
tail -f breakthrough_log.txt
```

### Quick Status Check:
```bash
# Get current best architecture with score in one line
curl -s "http://localhost:8001/candidates/top-k/1" | jq -r '.[0] | "\(.name): \(.score) (at \(.time))"'

# Alternative detailed format
BEST=$(curl -s "http://localhost:8001/candidates/top-k/1" | jq -r '.[0]')
echo "Best Architecture: $(echo $BEST | jq -r '.name')"
echo "Score: $(echo $BEST | jq -r '.score')"
echo "Time: $(echo $BEST | jq -r '.time')"

# Check multiple recent architectures with scores
curl -s "http://localhost:8001/elements/top-k/5" | jq -r '.[] | "\(.name): \(.score // "no score") (\(.time))"'
```

### Manual Breakthrough Investigation:
```bash
# When you suspect a breakthrough, investigate:
ARCH_NAME="<suspected_breakthrough_name>"

# Get full architecture details
curl -X GET "http://localhost:8001/elements/by-name/$ARCH_NAME" | jq '.' > investigation.json

# Check training results
grep -A 5 -B 5 "$ARCH_NAME" pipeline/files/analysis/loss.csv
grep -A 5 -B 5 "$ARCH_NAME" pipeline/files/analysis/benchmark.csv

# Examine source code
cat "pipeline/pool/${ARCH_NAME}.py"
```

## 9. Troubleshooting Performance Regression

If new architectures are scoring lower than baseline (2.495), check these areas:

### Diagnostic Commands:
```bash
# Check if new architectures are being generated properly
ls -lt pipeline/pool/*.py | head -5

# Compare recent architecture scores to baseline
echo "=== Recent Architecture Scores ==="
curl -s "http://localhost:8001/elements/top-k/10" | jq -r '.[] | "\(.name): \(.score // "no score")"' | head -10
echo "=== Baseline Comparison ==="
echo "hybrid_linear_hrm: 2.495 (baseline)"

# Check if evolution is stuck in local minima
curl -s "http://localhost:8001/elements" | jq '.[] | .name' | sort | uniq -c | sort -nr | head -10
```

### Common Issues:
- **Identical Low Scores**: Multiple architectures with same low score (1.5) indicates evolution system issues
- **Training Failures**: Check `pipeline/files/debug/training_error.txt` for training issues
- **RAG Service**: Ensure papers are being fetched properly from `http://localhost:13142`
- **Candidate System**: Only 1 candidate suggests evolution isn't adding successful architectures

### Reset Strategies:
```bash
# If stuck, clear database and restart with fresh candidate
curl -X DELETE "http://localhost:8001/elements"
curl -X DELETE "http://localhost:8001/candidates"

# Add baseline candidate back
# (Check candidate_storage.json and use database API to re-add)
```

## 10. Success Indicators Summary

### 🚨 Immediate Alert Triggers:
- Score jumps >0.5 points from previous best
- Reasoning accuracy >0.85
- Training converges in <50% normal time
- Novel architectural patterns detected

### 📊 Trend Analysis:
- Consistent upward score trajectory
- Reduced training instability
- Better generalization metrics
- Innovative fusion of research concepts

### 💾 Auto-Backup Conditions:
- Score >3.0 (first significant breakthrough)
- Score >3.5 (major advancement)  
- Score >4.0 (paradigm shift)
- Novel architecture categories

Remember: The pipeline runs continuously, so breakthroughs can happen at any time. Set up monitoring to catch them as they occur!