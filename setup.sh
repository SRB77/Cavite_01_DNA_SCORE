#!/usr/bin/env bash
# =============================================================================
# Developer DNA Matrix — Phase 3 Setup & Reproducibility Script
# Usage: bash setup.sh
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Developer DNA Matrix — Phase 3 Setup               ║"
echo "║   Full reproducible pipeline                         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Python version check ──────────────────────────────────────────────
echo -e "${CYAN}[1/5] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED="3.10"
if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo -e "  ${GREEN}✓ Python ${PYTHON_VERSION} — OK${NC}"
else
    echo -e "  ${RED}✗ Python ${PYTHON_VERSION} found, but 3.10+ required.${NC}"
    echo "  Install from https://python.org and re-run setup.sh"
    exit 1
fi

# ── 2. Virtual environment ───────────────────────────────────────────────
echo -e "${CYAN}[2/5] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "  ${GREEN}✓ Created venv/${NC}"
else
    echo -e "  ${YELLOW}→ venv/ already exists, skipping creation${NC}"
fi

# Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || {
    echo -e "${RED}  ✗ Could not activate venv. Try manually: source venv/bin/activate${NC}"
    exit 1
}
echo -e "  ${GREEN}✓ Virtual environment activated${NC}"

# ── 3. Install dependencies ──────────────────────────────────────────────
echo -e "${CYAN}[3/5] Installing dependencies from requirements.txt...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "  ${GREEN}✓ All dependencies installed${NC}"

# ── 4. Create output directories ─────────────────────────────────────────
echo -e "${CYAN}[4/5] Preparing output directories...${NC}"
mkdir -p outputs data figures reports
echo -e "  ${GREEN}✓ outputs/ data/ figures/ reports/ ready${NC}"

# ── 5. Run Phase 3 pipeline ──────────────────────────────────────────────
echo -e "${CYAN}[5/5] Running Phase 3 pipeline (this takes ~2 min)...${NC}"
echo ""
python phase3_pipeline.py

echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ✓ Phase 3 pipeline complete!                       ║"
echo "║                                                      ║"
echo "║   Outputs saved to outputs/                          ║"
echo "║     • phase3_confusion_matrix.png                    ║"
echo "║     • cross_phase_comparison.png                     ║"
echo "║     • continuous_score_distribution.png              ║"
echo "║     • feature_importance_phase3.png                  ║"
echo "║     • phase3_continuous_output.csv                   ║"
echo "║     • phase3_results.csv                             ║"
echo "║                                                      ║"
echo "║   To open notebooks:                                 ║"
echo "║     jupyter notebook                                 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
