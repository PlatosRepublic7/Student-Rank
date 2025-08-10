# Student Ranker - Quick Start Guide

## What it does
Compares three algorithms for assigning students to companies based on preferences:
- **Fill First** (0): Fill companies sequentially
- **Rank First** (1): Prioritize by student rankings  
- **Best First** (2): Select best available student per company

*Data Quality: Each student ranks exactly their top 5 companies (1,2,3,4,5) with no duplicate rankings. Remaining companies are unranked.*

## How to run

1. **No installation needed** - Just download `StudentRanker.exe` (34.6 MB)
2. **Open Command Prompt**: Windows Key + R, type `cmd`, press Enter
3. **Navigate to file**: `cd "C:\Path\To\Your\Downloaded\File"`
4. **Run program**: `StudentRanker.exe`

## Common commands

**Basic run:**
```cmd
StudentRanker.exe --trials 25
```

**Custom settings (Recommended):**
```cmd
StudentRanker.exe --trials 50 --students 40 --companies 12 --output_file my_results.csv
```

**Test specific algorithms:**
```cmd
StudentRanker.exe --trials 25 --algorithms 0 2
```

## Key options
- `--trials 25` - Number of test runs
- `--students 30` - Students per trial (default: 30)  
- `--companies 10` - Companies per trial (default: 10)
- `--output_file results.csv` - Name of results file
- `--help` - Show all options

## Results
- Opens CSV file in Excel to see detailed results
- **Lower average ranking = better** (closer to 1st choice)
- **Higher satisfaction % = better** (more students happy)
- **Choice distribution**: Both raw counts and percentages for each ranking (1st, 2nd, 3rd, etc.)

*Example: `fill_first_students_got_choice_1_count` shows how many students got their 1st choice*

## Troubleshooting
- **"Program not found"**: Type `dir` to check you're in right folder
- **Permission error**: Run Command Prompt as Administrator
- **Antivirus warning**: File is safe - compiled Python programs sometimes trigger warnings
