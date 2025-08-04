# Student-Rank 🎓

This project provides a set of Python scripts for ranking students into company groups based on their preferences. The main script, `ranker2.py`, implements several algorithms for this purpose.

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

The main script to run the ranking process is `ranker2.py`. It takes several command-line arguments to customize its behavior.

### Command-line Arguments:

  * `--file`: **(Required)** The path to the input CSV file containing student preferences. The file must have a `.csv` extension.
  * `--type`: **(Required)** An integer (0, 1, or 2) that determines the selection algorithm to be used:
      * `0`: Fill First (ff)
      * `1`: Rank First (rf)
      * `2`: Best First (bf)
  * `--students`: **(Optional)** The number of students to process from the input file. If not specified, all students in the file will be processed.
  * `--companies`: **(Optional, default=10)** The number of companies to rank and form groups for.
  * `--output_path`: **(Optional)** The path where the output CSV file will be saved. If not provided, the file will be saved in the current directory.
  * `--suppress_terminal_output`: **(Optional, default=False)** If set to `True`, it will suppress the detailed output in the terminal during execution.

### Example:

```bash
python ranker2.py --file data/form_test_data_cc.csv --type 1 --students 30
```

This command will run the ranking algorithm on the first 30 students from `form_test_data_cc.csv` using the "Rank First" method and output the results to a file named `ranker2data_rf.csv`.

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