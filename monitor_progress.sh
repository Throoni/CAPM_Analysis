#!/bin/bash

echo "═══════════════════════════════════════════════════════════════════════"
echo "PORTFOLIO OPTIMIZATION PROGRESS MONITOR"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "PORTFOLIO OPTIMIZATION PROGRESS"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    
    # Check if process is running
    if pgrep -f "portfolio_optimization" > /dev/null; then
        echo "Status: ✅ RUNNING"
    else
        echo "Status: ⏹️  COMPLETED"
        echo ""
        echo "Final output:"
        tail -50 /tmp/portfolio_opt_progress.log 2>/dev/null
        break
    fi
    
    echo ""
    echo "Latest progress (last 25 lines):"
    echo "───────────────────────────────────────────────────────────────────────"
    tail -25 /tmp/portfolio_opt_progress.log 2>/dev/null | tail -20
    echo ""
    echo "───────────────────────────────────────────────────────────────────────"
    echo "Press Ctrl+C to stop monitoring (process will continue running)"
    echo ""
    
    sleep 3
done

