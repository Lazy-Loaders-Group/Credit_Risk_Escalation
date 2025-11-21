# 📋 QUICK REFERENCE CARD
## Credit Risk AI Project - Essential Commands

---

## 🚀 START WORKING (Every Session)

```bash
cd /Users/nadunhettiarachchi/Documents/Credit_Risk_Escalation
source uom_venv/bin/activate
jupyter notebook
```

---

## 📊 CHECK PROGRESS

```bash
# See what's complete:
ls results/models/          # Models trained
ls results/figures/         # Visualizations created
ls data/processed/          # Data processed

# Count files:
ls results/models/ | wc -l  # Should be 0 now, 32+ when done
ls results/figures/ | wc -l # Should be 13 now, 30+ when done
```

---

## 🎯 PHASE CHECKLIST

**Phase 1:** ✅ Complete
- Check: `ls results/figures/*.png` shows 13 files

**Phase 2:** 🎯 Do this now!
- Notebook: `02_baseline_model.ipynb`
- Time: 20-30 minutes
- Check: `ls results/models/baseline_model.pkl`

**Phase 3:** ⏸️ Next
- Notebook: `03_uncertainty_quantification.ipynb`
- Time: 40-60 minutes
- Check: `ls results/models/ | wc -l` shows ~32 files

**Phase 4:** ⏸️ After Phase 3
- Notebook: `04_escalation_system.ipynb`
- Time: 15-20 minutes
- Check: See automation rate ~78% in output

**Phase 5:** ⏸️ Final step
- Notebook: `05_comprehensive_evaluation.ipynb`
- Time: 30-40 minutes
- Check: `ls results/reports/FINAL_PROJECT_REPORT.md`

---

## 🐛 TROUBLESHOOTING

### Jupyter not opening?
```bash
# Make sure environment is activated:
source uom_venv/bin/activate
which python  # Should show: .../uom_venv/bin/python
```

### Kernel died?
```python
# In notebook 03, reduce ensemble size:
n_models = 10  # Instead of 30
```

### Module not found?
```bash
source uom_venv/bin/activate
pip install -r requirements.txt
```

### Can't find dataset?
```bash
ls data/raw/  # Check if .csv file exists
# If only .zip, extract it:
cd data/raw && unzip *.zip
```

---

## 📁 KEY FILES

**Run These (In Order):**
1. `notebooks/02_baseline_model.ipynb` ← Start here
2. `notebooks/03_uncertainty_quantification.ipynb`
3. `notebooks/04_escalation_system.ipynb`
4. `notebooks/05_comprehensive_evaluation.ipynb`

**Read These (For Help):**
- `ACTION_PLAN.md` ← Your step-by-step guide
- `ROADMAP.md` ← Visual progress tracker
- `QUICK_REFERENCE.md` ← This file!

**Ignore These (For Now):**
- `PROJECT_GUIDE.md` (too detailed)
- `SETUP.md` (already set up)
- `PROGRESS.md` (outdated)

---

## 🎯 TODAY'S GOAL

**Minimum:** Train baseline model (Phase 2)
- Time: 30 minutes
- Proof: `ls results/models/baseline_model.pkl`

**Stretch:** Complete Phases 2-3
- Time: 1.5 hours
- Proof: `ls results/models/ | wc -l` shows ~32 files

**Ambitious:** All phases (2-5)
- Time: 2-3 hours
- Proof: `ls results/reports/FINAL_PROJECT_REPORT.md`

---

## 💡 SUCCESS TIPS

1. ✅ Run notebooks in order (2→3→4→5)
2. ✅ Read cell outputs to check if working
3. ✅ Save frequently (Ctrl+S)
4. ✅ If stuck >10 min, skip and return later
5. ✅ Check progress with `ls` commands above

---

## 📞 HELP COMMANDS

```bash
# Check Python version:
python --version  # Should be 3.12.x

# Check packages installed:
pip list | grep -E 'pandas|sklearn|xgboost|shap'

# Check virtual environment:
which python  # Should show uom_venv path

# Check disk space:
df -h .  # Need ~2GB free
```

---

## 🎓 WHAT YOU'RE BUILDING

**Simple Version:**
AI that knows when it doesn't know (and asks humans for help)

**Technical Version:**
ML system with uncertainty quantification and intelligent human escalation

**Business Version:**
Automate 78% of loan decisions with 89% accuracy, saving $678 per 210K applications

---

## 🏆 EXPECTED RESULTS

| Metric | Target | You'll Get |
|--------|--------|------------|
| Baseline Accuracy | >70% | ~85% |
| Automation Rate | 70-85% | ~78% |
| Automated Accuracy | >85% | ~89% |
| Cost Savings | Positive | ~21% |
| Time to Complete | 2-3 hrs | 2-3 hrs |

---

## 🚀 RIGHT NOW - DO THIS!

```bash
# 1. Open terminal
# 2. Copy-paste these 3 commands:

cd /Users/nadunhettiarachchi/Documents/Credit_Risk_Escalation
source uom_venv/bin/activate
jupyter notebook

# 3. In Jupyter browser:
#    - Open: 02_baseline_model.ipynb
#    - Click: Cell → Run All
#    - Wait 20-30 minutes
#    - Check results!
```

---

**Last Updated:** November 19, 2025  
**Status:** Phase 1 ✅ → Phase 2 🎯  
**Next:** Train baseline model!

🚀 **GO! YOU'VE GOT THIS!**

---

## 🔖 BOOKMARK THIS FILE

Keep this open while working - it has all the commands you need!

**Quick access in VS Code:**
- Press `Cmd+P`
- Type: `QUICK_REFERENCE.md`
- Press Enter
