#!/bin/bash

# Progress monitor for portfolio optimization
LOG_FILE="/tmp/portfolio_opt_progress.log"

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "PORTFOLIO OPTIMIZATION - LIVE PROGRESS MONITOR"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Check if process is running
    if pgrep -f "portfolio_optimization" > /dev/null; then
        echo "Status: ✅ RUNNING"
        PID=$(pgrep -f "portfolio_optimization" | head -1)
        echo "Process ID: $PID"
    else
        echo "Status: ⏹️  COMPLETED"
        echo ""
        echo "Final output:"
        tail -50 "$LOG_FILE" 2>/dev/null
        echo ""
        echo "═══════════════════════════════════════════════════════════════════════"
        break
    fi
    
    echo ""
    echo "Latest Progress (last 25 lines):"
    echo "───────────────────────────────────────────────────────────────────────"
    if [ -f "$LOG_FILE" ]; then
        tail -25 "$LOG_FILE" 2>/dev/null | tail -20
    else
        echo "Waiting for log file..."
    fi
    echo ""
    echo "───────────────────────────────────────────────────────────────────────"
    echo "Press Ctrl+C to stop monitoring (process will continue running)"
    echo "View full log: tail -f $LOG_FILE"
    echo ""
    
    sleep 3
done

