#!/bin/bash

# Ralph Wiggum - Autonomous AI Agent Loop
# Runs Claude Code CLI repeatedly until all PRD stories are complete

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
PROMPT_FILE="$SCRIPT_DIR/prompt.md"
ARCHIVE_DIR="$SCRIPT_DIR/.ralph-archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.ralph-last-branch"

# Configuration
MAX_ITERATIONS=${1:-25}
SLEEP_BETWEEN_ITERATIONS=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required files exist
check_prerequisites() {
    if [[ ! -f "$PRD_FILE" ]]; then
        log_error "PRD file not found: $PRD_FILE"
        exit 1
    fi

    if [[ ! -f "$PROMPT_FILE" ]]; then
        log_error "Prompt file not found: $PROMPT_FILE"
        exit 1
    fi

    if ! command -v claude &> /dev/null; then
        log_error "Claude Code CLI not found. Please install it first."
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log_error "jq not found. Please install it: sudo apt install jq"
        exit 1
    fi
}

# Get current branch from PRD
get_current_branch() {
    jq -r '.branch // "main"' "$PRD_FILE"
}

# Archive previous run if branch changed
handle_branch_change() {
    local current_branch=$(get_current_branch)

    if [[ -f "$LAST_BRANCH_FILE" ]]; then
        local last_branch=$(cat "$LAST_BRANCH_FILE")

        if [[ "$current_branch" != "$last_branch" ]]; then
            log_warning "Branch changed from $last_branch to $current_branch"

            mkdir -p "$ARCHIVE_DIR"
            local timestamp=$(date +%Y%m%d_%H%M%S)
            local archive_name="${last_branch}_${timestamp}"

            if [[ -f "$PROGRESS_FILE" ]]; then
                mv "$PROGRESS_FILE" "$ARCHIVE_DIR/progress_${archive_name}.txt"
                log_info "Archived progress file"
            fi
        fi
    fi

    echo "$current_branch" > "$LAST_BRANCH_FILE"
}

# Initialize progress file if it doesn't exist
init_progress_file() {
    if [[ ! -f "$PROGRESS_FILE" ]]; then
        cat > "$PROGRESS_FILE" << EOF
# Progress Log - Q for Mortals RAG Agent
# Started: $(date -Iseconds)

## Codebase Patterns
<!-- Consolidate reusable patterns here as they're discovered -->

---

EOF
        log_info "Created progress file"
    fi
}

# Count completed and total stories
get_progress() {
    local total=$(jq '.userStories | length' "$PRD_FILE")
    local completed=$(jq '[.userStories[] | select(.passes == true)] | length' "$PRD_FILE")
    echo "$completed/$total"
}

# Check if all stories are complete
all_stories_complete() {
    local incomplete=$(jq '[.userStories[] | select(.passes == false)] | length' "$PRD_FILE")
    [[ "$incomplete" -eq 0 ]]
}

# Main loop
main() {
    check_prerequisites
    handle_branch_change
    init_progress_file

    log_info "Starting Ralph Wiggum Agent Loop"
    log_info "PRD: $PRD_FILE"
    log_info "Max iterations: $MAX_ITERATIONS"
    log_info "Current progress: $(get_progress)"

    echo ""
    echo "============================================"
    echo "  RALPH WIGGUM - Autonomous Agent Loop"
    echo "============================================"
    echo ""

    for ((i=1; i<=MAX_ITERATIONS; i++)); do
        log_info "=== Iteration $i of $MAX_ITERATIONS ==="
        log_info "Progress: $(get_progress)"

        # Check if already complete
        if all_stories_complete; then
            log_success "All stories complete! Exiting."
            exit 0
        fi

        # Run Claude Code with the prompt
        log_info "Spawning Claude Code instance..."

        local output
        output=$(cat "$PROMPT_FILE" | claude --dangerously-skip-permissions 2>&1) || true

        # Check for completion signal
        if echo "$output" | grep -q "<promise>COMPLETE</promise>"; then
            log_success "Received COMPLETE signal from agent"

            if all_stories_complete; then
                log_success "All stories verified complete! Project finished."
                exit 0
            fi
        fi

        # Log iteration result
        echo "" >> "$PROGRESS_FILE"
        echo "### Iteration $i - $(date -Iseconds)" >> "$PROGRESS_FILE"
        echo "Progress: $(get_progress)" >> "$PROGRESS_FILE"

        # Brief pause between iterations
        if [[ $i -lt $MAX_ITERATIONS ]]; then
            log_info "Sleeping $SLEEP_BETWEEN_ITERATIONS seconds before next iteration..."
            sleep $SLEEP_BETWEEN_ITERATIONS
        fi
    done

    log_error "Reached maximum iterations ($MAX_ITERATIONS) without completing all stories"
    log_info "Final progress: $(get_progress)"
    exit 1
}

main "$@"
