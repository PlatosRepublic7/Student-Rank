# For argument parsing
import argparse

# For fake student names
from faker import Faker

# For fake company names

# General module imports
import random
import json

# Pandas and numpy
import pandas as pd
import numpy as np

def gen_random_student_ranking(num_students: int) -> list:
    fake = Faker()
    out_list = []
    r_list = []
    for _ in range(num_students):
        s_label = fake.name()
        r_list = random.sample(range(1, 11), 10)
        s_dict = {}
        s_dict[s_label] = r_list
        out_list.append(s_dict)
    return out_list


def clean_company_names(company_list: list) -> list:
    for name in company_list:
        name = str(name)
        name.replace("'", '')
        name.replace('"', '')
    return company_list


def gen_random_companies(num_companies=10) ->list:
    fake = Faker("en_US")
    company_names = [fake.company() for _ in range(num_companies)]
    out_list = clean_company_names(company_names)
    return out_list


def get_companies_from_file(file: str, num_companies: int) -> list:
    out_list = []
    with open(file) as f:
        for _ in range(num_companies):
            company_name = f.readline()
            company_name = company_name.rstrip('\n')
            out_list.append(company_name)

    return out_list


def gen_random_student_ranking_csv(num_students: int) -> list:
    fake = Faker()
    out_list = []
    r_list = []
    for _ in range(num_students):
        r_list = random.sample(range(1, 11), 10)
        out_list.append([fake.name()] + r_list)
    
    return out_list


def gen_wide_student_rankings_csv(num_students: int) -> list:
    fake = Faker()
    out_list = []
    for _ in range(num_students):
        r_list = [None for x in range(31)]
        r_list[0] = fake.name()
        random_index_numbers = np.random.choice(np.arange(1, 31), size=5, replace=False)
        for i in range(5):
            r_list[random_index_numbers[i]] = int(i + 1)
        out_list.append(r_list)

    return out_list        


def main():
    # Create argument parser and parse arguments
    parser = argparse.ArgumentParser(description="Program for Producing Student Rankings of Companies")
    parser.add_argument("--students", type=int, default=500)
    parser.add_argument("--companies", type=int, default=10)
    parser.add_argument("--file_type", type=str, default="csv", choices=["json", "csv"], help="Output File Type")
    parser.add_argument("--make_wide", type=bool, default=False, help="Flag for making a 30 company file with random rankings between 1 and 5")
    args = parser.parse_args()

    # Generating fake company names tends to give very poor results with Faker
    #company_list = gen_random_companies(args.companies)

    # Instead we generated them elsewhere, and will use a file as a bank of names to grab from
    if args.companies > 30:
        return f"Arg '--companies {args.companies}' exceeded maximum. Must be 30 or less."
    
    if args.make_wide:
        company_list = get_companies_from_file("./utilities/companies.txt", 30)
    else:
        company_list = get_companies_from_file('companies.txt', args.companies)

    if args.file_type == "json":
        student_list = gen_random_student_ranking(args.students)
        data = {
            'Companies': company_list,
            'Students': student_list
        }

        file_name = "test_data.json"

        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)

        return
    elif args.file_type == "csv":
        if not args.make_wide:
            # Generate student names and rankings in the appropriate format for a csv
            student_rankings = gen_random_student_ranking_csv(args.students)
            file_name = "test_data.csv"
        else:
            student_rankings = gen_wide_student_rankings_csv(args.students)
            file_name = "test_data_wide.csv"
        
        # Create the columns for the csv, with "Student" as the first column, and all the rankings following
        columns = ["Student"] + company_list

        # Create a DataFrame
        df = pd.DataFrame(student_rankings, columns=columns)

        for column in columns:
            if column == "Student":
                continue
            else:
                df[column] = df[column].astype('Int64')
        
        df.to_csv(file_name, index=False)
        return
    else:
        print(f"Invalid file type '{args.file_type}': Must be either 'json' or 'csv'")
        return


if __name__ == '__main__':
    main()
