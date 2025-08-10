# Student-Rank 🎓

This project provides a set of Python scripts for ranking students into company groups based on their preferences. The main script, `ranker2.py`, implements several algorithms for this purpose and can run in two modes:

1. **Trial Mode** (NEW): Generates synthetic data and runs statistical analysis across multiple trials
2. **Legacy Mode**: Processes existing CSV files with student preference data

-----

## Requirements

Before running the project, ensure you have the necessary dependencies installed. You can install them using pip and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The required packages are:

  * Faker==33.1.0
  * numpy==2.2.1
  * pandas==2.2.3
  * python-dateutil==2.9.0.post0
  * pytz==2024.2
  * six==1.17.0
  * typing\_extensions==4.12.2
  * tzdata==2024.2

-----

## Usage

### Trial Mode (Statistical Analysis)

The primary way to use this program is in **Trial Mode**, which generates synthetic student ranking data and runs statistical comparisons of the three assignment algorithms.

#### Command-line Arguments:

  * `--trials`: **(Optional, default=1)** Number of trials to run (maximum: 1000). Each trial uses freshly generated synthetic data.
  * `--students`: **(Optional, default=30)** Number of students per trial.
  * `--companies`: **(Optional, default=10)** Number of companies per trial.
  * `--algorithms`: **(Optional, default=all)** Which algorithms to test. Choose from:
      * `0`: Fill First (ff)
      * `1`: Rank First (rf) 
      * `2`: Best First (bf)
      * Can specify multiple: `--algorithms 0 2` for Fill First and Best First only
  * `--output_file`: **(Optional, default="algorithm_statistics.csv")** Name of the output CSV file.
  * `--suppress_terminal_output`: **(Optional)** Suppress detailed progress output during trials.
  * `--progress_interval`: **(Optional, default=10)** Show progress every N trials.

#### Examples:

```bash
# Run 100 trials comparing all three algorithms
python ranker2.py --trials 100 --students 50 --companies 10

# Compare only Fill First vs Best First algorithms across 500 trials
python ranker2.py --trials 500 --algorithms 0 2 --output_file "ff_vs_bf_comparison.csv"

# Large-scale analysis with 1000 trials and progress updates every 50 trials
python ranker2.py --trials 1000 --students 100 --companies 15 --progress_interval 50
```

#### Output Format:

The trial mode produces a CSV file where each row represents one trial, and columns contain statistical metrics for each algorithm:

- `trial`: Trial number
- `{algorithm}_mean_ranking`: Average ranking assigned to students (lower is better)
- `{algorithm}_satisfaction_rate`: Percentage of students receiving ranks 1-3
- `{algorithm}_group_size_variance`: Variance in group sizes (lower means more even distribution)
- `{algorithm}_rank_{n}_pct`: Percentage of students receiving rank n

Where `{algorithm}` is `ff` (Fill First), `rf` (Rank First), or `bf` (Best First).

### Legacy Mode (Single File Processing)

For backward compatibility, you can still process existing CSV files using legacy mode.

#### Legacy Command-line Arguments:

  * `--file`: **(Required for legacy mode)** The path to the input CSV file containing student preferences. The file must have a `.csv` extension.
  * `--type`: **(Required for legacy mode)** An integer (0, 1, or 2) that determines the selection algorithm to be used:
      * `0`: Fill First (ff)
      * `1`: Rank First (rf)
      * `2`: Best First (bf)
  * `--students`: **(Optional)** The number of students to process from the input file. If not specified, all students in the file will be processed.
  * `--companies`: **(Optional, default=10)** The number of companies to rank and form groups for.
  * `--suppress_terminal_output`: **(Optional)** Suppress detailed terminal output during execution.

#### Legacy Example:

```bash
python ranker2.py --file data/form_test_data_cc.csv --type 1 --students 30
```

This command will run the ranking algorithm on the first 30 students from `form_test_data_cc.csv` using the "Rank First" method and output the results to a file named `ranker2data_rf.csv`.

-----

## Algorithm Comparison

The program implements three different student assignment algorithms:

### Fill First (Type 0)
- **Strategy**: Assigns students company-by-company, starting with the lowest-ranked companies
- **Characteristics**: Tends to provide good overall satisfaction but may create uneven group distributions
- **Best for**: Scenarios where ensuring every company gets good students is important

### Rank First (Type 1) 
- **Strategy**: Processes all rank-1 preferences first, then rank-2, etc., across all companies simultaneously
- **Characteristics**: Prioritizes top preferences but may leave some students with poor assignments
- **Best for**: Maximizing the number of students who get their top choices

### Best First (Type 2)
- **Strategy**: Iteratively finds the best available student-company match across all companies
- **Characteristics**: Most computationally complex, aims for optimal overall assignment
- **Best for**: Balancing individual satisfaction with overall optimization

-----

-----

## Input File Format

The script requires a specific CSV format for the input file. The `cparser.py` script is responsible for parsing this format.

  * The CSV file must have a header row.
  * The first three columns should be `Timestamp`, `Email`, and `Name`.
  * Subsequent columns represent the companies. The header for each company column must be in the format: `"CompanyName - ProposerEmail@example.com"`.
  * The values in the company columns represent the rank given by the student to that company (e.g., 1, 2, 3, 4, 5).
  * If a student has not ranked a company, the corresponding cell should be empty. These unranked companies are treated as having a rank of 6.

An example of this format can be seen in the provided `data/form_test_data_cc.csv` file.

-----

## Output File Format

The script will generate a CSV file with the final student groupings. The name of the output file depends on the chosen algorithm:

  * `ranker2data_ff.csv` for Fill First
  * `ranker2data_rf.csv` for Rank First
  * `ranker2data_bf.csv` for Best First

The output file will have the following columns:

  * `Company`: The name of the company.
  * `Student X`: The name of the student assigned to the group.
  * `Student X Rank`: The rank the student gave to that company.

-----

## Utilities

The `utilities` directory contains several helper scripts:

  * **`form_data_generator.py`**: This script can be used to generate test data in the required CSV format. It uses `companies.txt` and `names_and_emails.txt` to create the data.
  * **`gen_s_data.py`**: A more general-purpose script for generating fake student ranking data in either JSON or CSV format.
  * **`analyze_data.py`**: This script can be used to analyze a CSV file of student rankings and count the number of times each rank (1-10) was given to each company. It outputs the results to `test_data_counts.csv`.
  * **`companies.txt`**: A simple text file containing a list of company names, one per line.
  * **`names_and_emails.txt`**: A text file containing student names and their corresponding email addresses, separated by " - ".
  * **`extra_names_and_emails.txt`**: An additional list of names and emails.