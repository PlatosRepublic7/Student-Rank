# Student-Rank 🎓

A comprehensive Python-based student assignment system that optimally allocates students to company groups based on preference rankings. This program implements three distinct assignment algorithms with statistical analysis capabilities, designed for scenarios like internship placements, project assignments, or group formations.

## 🎯 Overview

The Student-Rank system addresses the complex problem of fairly assigning students to companies/groups when both parties have preferences. The program:

- **Generates realistic synthetic data** with proper ranking constraints (unique ranks 1-5 per student)
- **Implements three assignment algorithms** with different optimization strategies
- **Provides comprehensive statistical analysis** with multiple performance metrics
- **Supports both batch analysis and single-file processing**
- **Includes a standalone executable** for non-technical users

## 🚀 Quick Start

### For Technical Users
```bash
# Install dependencies
pip install -r requirements.txt

# Run statistical comparison of all algorithms
python ranker2.py --trials 100 --students 50 --companies 10

# Generate standalone executable
pyinstaller --onefile --name StudentRanker ranker2.py
```

### For Non-Technical Users
Use the pre-built `StudentRanker.exe` executable:
```bash
StudentRanker.exe --trials 50 --students 30 --companies 8
```

## 📋 Requirements

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `pandas>=2.2.3` - Data manipulation and analysis
- `numpy>=2.2.1` - Numerical computing and vectorized operations
- `Faker>=33.1.0` - Synthetic data generation
- `tqdm>=4.66.0` - Progress bars (optional, enhanced UX)

**Development Dependencies:**
- `PyInstaller>=6.15.0` - Standalone executable creation

## 🔧 System Architecture

### Core Components

#### 1. **Data Generation Engine** (`generate_synthetic_data`)
- **Purpose:** Creates realistic student preference data with proper constraints
- **Features:**
  - Ensures unique rankings 1-5 per student (no duplicates)
  - Remaining companies automatically ranked as 6 (unranked/equal weight)
  - Uses company names from file or generates fake companies
  - Configurable student and company counts

#### 2. **Company Ranking System** (`get_ranked_companies`)
- **Purpose:** Evaluates and ranks companies based on student preferences
- **Algorithm:**
  - **Weighted Scoring:** Rank weights [5,4,3,2,1] for ranks 1-5
  - **Composite Score:** 75% rank score + 25% adjusted score  
  - **Tie-Breaking:** Secondary sort by Rank 1, Rank 2, etc. counts
  - **Final Selection:** Random selection among perfect ties

#### 3. **Assignment Algorithms** (Three distinct strategies)
- **Fill First:** Company-centric sequential optimization
- **Rank First:** Student preference-centric rank-by-rank processing  
- **Best First:** Iterative quality-focused selection

#### 4. **Statistical Analysis Engine** (`collect_algorithm_statistics`)
- **Metrics Collected:**
  - Average student ranking and standard deviation
  - Student satisfaction rates with confidence intervals
  - Detailed choice distribution (counts and percentages)
  - Group size variance and distribution analysis

#### 5. **Progress & Output Management**
- **Real-time Progress:** tqdm-based progress bars (default enabled)
- **Comprehensive CSV Output:** Raw counts and percentage distributions
- **Terminal Output:** Color-coded results with statistical summaries

## 📊 Assignment Algorithms Deep Dive

### 1. Fill First Algorithm (`get_student_groups`)
**Philosophy:** Company-centric optimization with sequential processing

**Strategy:**
- Companies processed in reverse ranking order (best companies last)
- Each company group filled completely before moving to next
- Greedy group sizing approach for early allocation

**Selection Process:**
1. **Proposed Company Priority:** Students who proposed this company get automatic selection
2. **Rank-Based Selection:** Lower ranks (higher preference) prioritized
3. **Tie-Breaking:** Higher "remaining rank average" preferred
4. **Complete Filling:** Entire group filled before next company

**Characteristics:**
- ✅ **Guarantees:** Every company gets filled to capacity
- ✅ **Consistency:** Predictable, deterministic results
- ⚠️ **Trade-off:** Early companies may get better students
- 📈 **Performance:** Fast execution, simple logic

### 2. Rank First Algorithm (`get_rank_first_student_groups`)
**Philosophy:** Student preference-centric with democratic processing

**Strategy:**
- Process all rank-1 preferences first, then rank-2, etc.
- All companies compete simultaneously for students at each rank level
- Progressive filling ensures top preferences honored across all companies

**Selection Process:**
1. **Rank-Level Processing:** All students with rank N processed simultaneously
2. **Forced Placement:** Proposed companies get automatic student placement
3. **Competitive Selection:** Companies compete for remaining students
4. **Progressive Ranks:** Continue through ranks 1→6 until all groups filled

**Characteristics:**
- ✅ **Student-Focused:** Maximizes fulfillment of top preferences
- ✅ **Democratic:** All companies get equal opportunity at each rank
- ✅ **Fair Distribution:** Prevents early companies from monopolizing good students
- 📈 **Performance:** Excellent student satisfaction rates

### 3. Best First Algorithm (`get_best_first_student_groups`)
**Philosophy:** Quality optimization with balanced iterative selection

**Strategy:**
- Each round, every company selects their single best available student
- Continues until all groups are filled
- Balances individual quality with overall fairness

**Selection Process:**
1. **Round-Based:** Each company gets one selection per round
2. **Quality Criteria:** 
   - Automatic selection for proposed company + rank 1 match
   - Otherwise: best combination of low rank + high remaining average
3. **Iterative Filling:** Continues rounds until all positions filled
4. **Balanced Opportunity:** Each company participates in every round

**Characteristics:**
- ✅ **Quality-Focused:** Optimizes for best possible matches
- ✅ **Balanced:** Equal selection opportunity per round
- ⚠️ **Variability:** Results can vary based on competition dynamics
- 📈 **Performance:** Sophisticated optimization, longer execution

## 📈 Statistical Analysis Framework

### Primary Metrics

#### Student Satisfaction Analysis
- **Average Ranking:** Mean rank assigned to students (1.0 = perfect)
- **Standard Deviation:** Consistency of assignments
- **Satisfaction Rate:** Percentage receiving ranks 1-3
- **Choice Distribution:** Detailed breakdown of rank 1-6 assignments

#### Group Formation Analysis  
- **Group Size Variance:** Evenness of group distribution
- **Company Ranking:** Quality assessment of company selections
- **Assignment Efficiency:** Speed and resource utilization

### Output Format
**Enhanced CSV Structure:**
```csv
trial,algorithm_mean_ranking,algorithm_std_dev,algorithm_satisfaction_rate,
algorithm_rank_1_count,algorithm_rank_1_pct,algorithm_rank_2_count,algorithm_rank_2_pct,
[...continues for all ranks]
```

**Statistical Summary:**
- Cross-trial averages with confidence intervals
- Algorithm performance comparisons
- Variance and consistency analysis

-----

## 💻 Usage Guide

### Trial Mode (Statistical Analysis) - Primary Use Case

The main functionality generates synthetic data and performs statistical comparisons across multiple trials.

#### Core Arguments:
- `--trials` **(Optional, default=1)** - Number of trials (max: 1000)
- `--students` **(Optional, default=30)** - Students per trial  
- `--companies` **(Optional, default=10)** - Companies per trial
- `--algorithms` **(Optional, default=all)** - Algorithm selection:
  - `0`: Fill First (`ff`)
  - `1`: Rank First (`rf`)
  - `2`: Best First (`bf`)
  - Multiple: `--algorithms 0 2` for Fill First + Best First only
- `--output_file` **(Optional, default="algorithm_statistics.csv")** - Output filename
- `--no_progress_bar` **(Optional)** - Disable progress indicators

#### Advanced Arguments:
- `--suppress_terminal_output` **(Optional)** - Minimize console output
- `--progress_interval` **(Optional, default=10)** - Progress update frequency

#### Example Commands:

```bash
# Standard comparison across all algorithms
python ranker2.py --trials 100 --students 50 --companies 10

# Large-scale analysis with specific algorithms
python ranker2.py --trials 500 --algorithms 0 2 --output_file "ff_vs_bf_analysis.csv"

# High-volume testing with progress tracking
python ranker2.py --trials 1000 --students 100 --companies 15 --progress_interval 50

# Executable version (same arguments)
./dist/StudentRanker.exe --trials 50 --students 25 --companies 8
```

#### Performance Characteristics:
- **Optimized Execution:** 3.5-9.7x speedup through vectorized operations
- **Memory Efficient:** Processes large datasets without memory issues
- **Progress Tracking:** Real-time feedback for long-running analyses
- **Error Handling:** Robust error handling with informative messages

### Legacy Mode (Single File Processing)

For processing existing CSV files with student preference data.

#### Legacy Arguments:
- `--file` **(Required)** - Path to input CSV file
- `--type` **(Required)** - Algorithm selection (0, 1, or 2)
- `--students` **(Optional)** - Number of students to process
- `--companies` **(Optional, default=10)** - Number of companies
- `--suppress_terminal_output` **(Optional)** - Minimize output

#### Legacy Example:
```bash
python ranker2.py --file data/form_test_data_cc.csv --type 1 --students 30
```

## 📊 Algorithm Performance Analysis

### Typical Performance Characteristics

Based on extensive testing across various scenarios:

#### Fill First Algorithm
- **Student Satisfaction:** 85-95% (high consistency)
- **Average Ranking:** 1.8-2.2 (good overall scores)
- **Strengths:** Reliable, predictable, company-friendly
- **Best For:** Scenarios prioritizing company satisfaction and predictability

#### Rank First Algorithm  
- **Student Satisfaction:** 88-96% (highest student satisfaction)
- **Average Ranking:** 1.7-2.0 (excellent student outcomes)
- **Strengths:** Maximizes student preference fulfillment
- **Best For:** Student-centric scenarios where top choices matter most

#### Best First Algorithm
- **Student Satisfaction:** 70-85% (variable, optimization-dependent)
- **Average Ranking:** 2.2-2.8 (more variable outcomes)
- **Strengths:** Theoretical optimization, balanced approach
- **Best For:** Complex scenarios requiring sophisticated matching

### Performance Metrics Interpretation

#### Statistical Significance
- **Mean Ranking:** Lower values indicate better student satisfaction
- **Standard Deviation:** Lower values indicate more consistent results
- **Satisfaction Rate:** Percentage receiving ranks 1-3 (target: >85%)
- **Choice Distribution:** Detailed breakdown shows preference fulfillment patterns

#### Comparative Analysis
The program enables direct comparison through:
- **Cross-algorithm trials** with identical synthetic data
- **Statistical significance testing** across multiple runs
- **Variance analysis** to assess consistency
- **Edge case testing** with extreme student/company ratios

## 🔧 Technical Implementation Details

### Data Generation Algorithm
```python
# Ranking Constraint Logic
for each student:
    - Select 5 random companies to rank
    - Assign unique ranks 1-5 to selected companies  
    - Set all other companies to rank 6 (unranked)
    - Ensures no duplicate rankings per student
```

### Company Scoring System
```python
# Weighted Scoring Formula
rank_score = Σ(rank_count[i] × weight[i]) for i in [1,2,3,4,5]
weights = [5, 4, 3, 2, 1]  # Rank 1 = 5 points, Rank 2 = 4 points, etc.

adjusted_score = rank_score / total_votes
composite_score = 0.75 × rank_score + 0.25 × adjusted_score
```

### Memory and Performance Optimizations
- **Vectorized Operations:** Pandas/NumPy optimizations for large datasets
- **Efficient Data Structures:** Minimal memory footprint for large trials
- **Progress Tracking:** Non-blocking progress updates
- **Error Recovery:** Graceful handling of edge cases

## 📁 File Structure & Components

### Core Files
- **`ranker2.py`** - Main program with all algorithms and analysis
- **`cparser.py`** - CSV parsing utilities for legacy mode
- **`requirements.txt`** - Python dependencies
- **`StudentRanker.exe`** - Standalone executable (generated)

### Data Directory (`data/`)
Contains test datasets and examples:
- **`form_test_data*.csv`** - Sample preference data files
- **`rank_data.json`** - JSON format test data
- **`rank_your_sheet.csv`** - Template for manual data entry

### Results Directory (`results/`)
Algorithm output examples and performance comparisons:
- **`*_first_*.csv`** - Algorithm-specific result files
- **`bf_bu_results.jpg`** - Performance visualization

### Utilities Directory (`utilities/`)
Helper scripts and configuration:
- **`form_data_generator.py`** - Synthetic data generation
- **`analyze_data.py`** - Data analysis utilities  
- **`companies.txt`** - Company name database
- **`names_and_emails.txt`** - Student identity database

-----

## 📊 Input/Output Specifications

### Input File Format (Legacy Mode)

For processing existing CSV files, the system expects a specific format:

#### CSV Structure Requirements:
- **Header Row:** Required with specific column names
- **Core Columns:** `Timestamp`, `Email`, `Name` (first three columns)
- **Company Columns:** Format: `"CompanyName - ProposerEmail@example.com"`
- **Ranking Values:** Integers 1-5 (student preferences)
- **Missing Rankings:** Empty cells treated as rank 6 (unranked)

#### Example Format:
```csv
Timestamp,Email,Name,TechCorp - john@example.com,DataSys - jane@example.com
2024-01-01,student1@school.edu,Alice Johnson,1,3
2024-01-01,student2@school.edu,Bob Smith,2,1
```

### Output File Formats

#### Trial Mode Output (`algorithm_statistics.csv`)
Comprehensive statistical analysis with the following structure:

```csv
trial,ff_mean_ranking,ff_std_dev,ff_satisfaction_rate,ff_median,ff_iqr,
ff_rank_1_count,ff_rank_1_pct,ff_rank_2_count,ff_rank_2_pct,...
[repeated for rf_ and bf_ algorithms]
```

**Column Definitions:**
- `{alg}_mean_ranking` - Average rank assigned to students
- `{alg}_std_dev` - Standard deviation of rankings  
- `{alg}_satisfaction_rate` - Percentage receiving ranks 1-3
- `{alg}_median` - Median ranking value
- `{alg}_iqr` - Interquartile range (Q3-Q1)
- `{alg}_rank_N_count` - Raw count of students receiving rank N
- `{alg}_rank_N_pct` - Percentage of students receiving rank N

#### Legacy Mode Output (`ranker2data_{algorithm}.csv`)
Student group assignments by company:

```csv
Company,Student 1,Student 1 Rank,Student 2,Student 2 Rank,...
TechCorp,Alice Johnson,1,Bob Smith,2,...
DataSys,Carol White,1,David Brown,3,...
```

## 🔬 Research Applications & Use Cases

### Academic Research
- **Educational Psychology:** Student preference satisfaction analysis
- **Operations Research:** Multi-objective optimization studies
- **Computer Science:** Algorithm performance comparison research
- **Statistics:** Large-scale randomized trial analysis

### Practical Applications
- **University Programs:** Internship and project assignments
- **Corporate Training:** Team formation and skill matching
- **Event Management:** Workshop and session assignments
- **Research Institutions:** Lab rotation assignments

### Comparative Studies
The system enables research into:
- **Fairness vs. Efficiency** trade-offs in assignment algorithms
- **Scale Effects** on algorithm performance (small vs. large groups)
- **Preference Distribution** impact on satisfaction outcomes
- **Real-world vs. Synthetic** data performance validation

## 🧮 Advanced Configuration

### Synthetic Data Generation Parameters

#### Student Preference Modeling
- **Ranking Distribution:** Uniform random selection of 5 companies to rank
- **Preference Realism:** No duplicate rankings per student
- **Company Proposal:** Random assignment of one proposed company per student
- **Scale Testing:** Supports 1-1000 students, 1-1000 companies

#### Company Name Generation
- **File-based:** Loads from `utilities/companies.txt`
- **Faker Integration:** Generates realistic company names when file insufficient
- **Uniqueness:** Ensures no duplicate company names in dataset

### Performance Tuning

#### Memory Optimization
```python
# Vectorized operations for large datasets
# Efficient DataFrame operations
# Minimal object creation in loops
```

#### Execution Speed
- **Parallel Processing:** Ready for multi-threading implementation
- **Caching:** Optimized data structure reuse
- **Progress Tracking:** Non-blocking progress updates

### Statistical Significance Testing

#### Confidence Intervals
The system calculates confidence intervals for:
- Mean ranking values
- Satisfaction rates  
- Standard deviations
- Choice distribution percentages

#### Sample Size Recommendations
- **Minimum Trials:** 30+ for basic significance
- **Recommended Trials:** 100+ for reliable comparisons
- **High-confidence Studies:** 500+ trials for publication-quality results

## 🚀 Development & Extension

### Code Architecture

#### Modular Design
- **Algorithm Functions:** Independent, swappable implementations
- **Data Generation:** Separate, configurable synthetic data creation
- **Statistical Analysis:** Comprehensive metrics collection
- **I/O Handling:** Flexible input/output format support

#### Extension Points
```python
# Adding new algorithms
algorithm_functions = {
    0: ('Fill First', get_student_groups),
    1: ('Rank First', get_rank_first_student_groups),
    2: ('Best First', get_best_first_student_groups),
    3: ('Custom Algorithm', your_algorithm_function)  # Add here
}
```

### Custom Algorithm Development

#### Required Function Signature
```python
def your_algorithm(df: pd.DataFrame, ttdf: pd.DataFrame, 
                  num_students: int, suppress_output: bool, 
                  proposed_companies: dict) -> pd.DataFrame:
    # Your algorithm implementation
    return student_groups_df
```

#### Integration Steps
1. Implement algorithm function with required signature
2. Add to `algorithm_functions` dictionary
3. Update command-line argument parsing
4. Test with existing statistical framework

### Performance Benchmarking

Current optimization achievements:
- **3.5x speedup** on small datasets (30 students, 10 companies)
- **9.7x speedup** on large datasets (100+ students, 20+ companies)
- **Memory efficiency** improvements for 1000+ trial runs
- **Progress tracking** with minimal performance impact

## 🤝 Contributing & Support

### Bug Reports
When reporting issues, please include:
- Command-line arguments used
- Error messages (full traceback)
- Operating system and Python version
- Sample data (if applicable)

### Feature Requests
Priority areas for enhancement:
- Additional assignment algorithms
- More sophisticated preference modeling
- Advanced visualization capabilities
- Multi-objective optimization features

### Development Setup
```bash
# Clone repository
git clone https://github.com/YourUsername/Student-Rank.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install pyinstaller  # for executable building
```

---

## 📚 References & Citations

### Algorithm Background
- **Assignment Problems:** Classical operations research literature
- **Preference Satisfaction:** Social choice theory and mechanism design
- **Statistical Analysis:** Educational psychology and satisfaction measurement

### Technical Implementation
- **Pandas/NumPy:** High-performance data analysis libraries
- **Faker:** Realistic synthetic data generation
- **PyInstaller:** Cross-platform executable packaging

---

*This documentation reflects the current state of the Student-Rank system as of August 2025. For the latest updates and additional features, please check the repository's latest commits.*