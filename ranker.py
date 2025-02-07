'''
ranker.py           VERSION 1.0
Ryan Kitson         January 2025
'''

# For argument parsing
import argparse

# General utility imports
from math import ceil
import random

# Pandas and Numpy
import pandas as pd
import numpy as np

# Text coloring class for use within Window's terminals
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# FUNCTIONS

def select_most_desirable_company(df: pd.DataFrame, suppress_terminal_output: bool, select_top_ten=False):
    '''
    Returns a Series corresponding to the company with the highest desirability. The desirability is 
    determined by taking the mean-corrected L^2 Norm of a vector composed of the companies mean-ranking, 
    and the standard deviation.
    '''
    # Get the mean and variance for all the company columns of the passed-in dataframe
    mean_series = df.mean(numeric_only=True)
    variance_series = df.var(numeric_only=True)
    median_series = df.median(numeric_only=True)
    skew_series = df.skew(numeric_only=True)
    sorted_mean = mean_series.sort_values()
    sorted_variance = variance_series.sort_values()
    sorted_median = median_series.sort_values()
    mv_df = pd.DataFrame({'Mean': sorted_mean, 'Variance': sorted_variance})

    # Create new columns. CL2_norm_... columns stand for "Mean-Corrected L^2 Norm" and indicate that
    # we shift the vectors coordinate over by one to correct for the lowest possible mean of 1.
    
    mv_df['Std_Dev'] = np.sqrt(mv_df['Variance'])
    mv_df['Median'] = median_series
    mv_df['Skew'] = skew_series

    #mv_df['L2_norm_var'] = np.sqrt(mv_df['Mean']**2 + mv_df['Variance']**2)
    #mv_df['CL2_norm_var'] = np.sqrt((mv_df['Mean'] - 1)**2 + mv_df['Variance']**2)
    #mv_df['L2_norm_std'] = np.sqrt(mv_df['Mean']**2 + mv_df['Std_Dev']**2)
    mv_df['CL2_Norm'] = np.sqrt((mv_df['Mean'] - 1)**2 + mv_df['Std_Dev']**2)
    
    # Experimental Desirability metric
    
    alpha = 0
    
    #mv_df['Des_experimental'] = np.sqrt((mv_df['Mean'] - 1)**2 + alpha * (mv_df['Mean'] + mv_df['Std_Dev'])**2 + (np.exp(-1 * beta * mv_df['Skew']))**2)
    
    mv_df['Des_Met'] = np.sqrt((mv_df['Mean'] - 1)**2 + (mv_df['Std_Dev'])**2 + (np.exp(-1 * alpha * mv_df['Skew']))**2)
    mv_df['Desirability'] = 100* (np.sqrt((mv_df['Mean'] - 1)**2 + (mv_df['Std_Dev'])**2 + (np.exp(-1 * alpha * mv_df['Skew']))**2))**(-1)
    # For the 3rd component, we're going to use a combination of the mean, median, and standard deviation
    # The forula is this: (mean - median)^2 / (1 + std_dev) or / (1 + variance)
    #mv_df['Des_std'] = (mv_df['Median'] - mv_df['Mean'])**2 / (1 + mv_df['Std_Dev'])
    #mv_df['Des_var'] = (mv_df['Median'] - mv_df['Mean'])**2 / (1 + mv_df['Variance'])

    #var_ranked_df = mv_df.sort_values(by='CL2_norm_var')
    #print(var_ranked_df)
    std_ranked_df = mv_df.sort_values(by='CL2_Norm')
    des_ranked_df = mv_df.sort_values(by='Des_Met', ascending=False)
    if not suppress_terminal_output:
        #print(std_ranked_df)
        print(des_ranked_df)
    #both_ranked_df = mv_df.sort_values(by='CL2_norm_both')
    #print(both_ranked_df)
    # Select the lowest CL2_norm_std
    if select_top_ten:
        des_ranked_df = des_ranked_df.sort_values(by='Des_Met', ascending=True)
        top_ten_companies = []
        count = 0
        for index, row in des_ranked_df.iterrows():
            top_ten_companies.append(index)
            count += 1
            if count > 9:
                break
        return top_ten_companies
    else:
        highest_ranked = des_ranked_df.iloc[0]
        return highest_ranked
    

def select_company_group(df: pd.DataFrame, num_slots: int) -> list:
    '''
    Returns a list of Student names selected as a group
    '''
    # Get the column with all the rankings
    column_names = df.columns
    rankings = df[column_names[1]]
    remaining_rank_averages = df[column_names[2]]

    # Generate a random starting index, starting max_rank, and dummy entries for our student group
    starting_index = random.randint(0, len(rankings) - 1)
    max_rank = 11
    student_names = [{f'NULL{k}': max_rank, 'average_remaining': 11} for k in range(num_slots)]
    current_group = []
    for k in range(num_slots):
        current_group.append([None, max_rank, max_rank])

    # We will iterate through the rankings, keeping track of the student it belongs to, making sure that
    # we apply our filtering logic
    # Perhaps we can use a list of lists: [['Student's name', 'rank', 'average_remaining'], ...]
    # And we return a list like this: [['Student's name', 'rank'], ...]
    for i in range(len(rankings)):
        index = (starting_index + i) % len(rankings)
        if rankings[index] <= max_rank:
            for j in range(len(current_group)):
                if current_group[j][1] == max_rank:
                    if rankings[index] < current_group[j][1] or remaining_rank_averages[index] > current_group[j][2]:
                        student_row = df.iloc[index]
                        student_dict = student_row.to_dict()
                        current_group[j][0] = student_dict['Student']
                        current_group[j][1] = student_dict[column_names[1]]
                        current_group[j][2] = student_dict[column_names[2]]
                        max_rank = get_list_max_rank(current_group)
                        current_group = sorted(current_group, key=lambda x: (x[1], x[2]))
                        break
                    
    final_group = [[], []]
    for i in range(len(current_group)):
        current_group[i].pop(2)
        final_group[0].append(current_group[i][0])
        final_group[1].append(current_group[i][1])
    
    return final_group


def get_top_ten_companies(df: pd.DataFrame):
    highest_ranked = select_most_desirable_company(df, True, True)
    return highest_ranked


def get_list_max_rank(in_list: list) -> int:
    max = 0
    for i in range(len(in_list)):
        if in_list[i][1] > max:
            max = in_list[i][1]
    return max


def get_max_rank(in_list: list) -> int:
    max = 0
    for i in range(len(in_list)):
        for key, value in in_list[i].items():
            if value > max:
                max = value
    return max


def make_list_from_student_dicts(in_list: list) -> list:
    out_list = [[], []]
    for i in range(len(in_list)):
        for name, rank in in_list[i].items():
            out_list[0].append(name)
            out_list[1].append(rank)
    return out_list


def calculate_company_buckets(num_students: int, num_companies: int) -> list:
    '''
    This will return a list of the number of available slots for each company. It progressively
    calculates the ceiling of num_students / num_companies, taking into account the remaining students
    and companies after each iteration. This ensures that an equitable distribution of students can be 
    assigned to each group, and prioritizes the more desirable companies first (lower list index).
    '''
    
    out_list = []
    for i in range(num_companies):
        num_slots = ceil(num_students / (num_companies - i))
        out_list.append(num_slots)
        num_students -= num_slots

    return out_list


def get_remaining_rank_averages(df: pd.DataFrame) -> list:
    out_list = []
    for index, row in df.iterrows():
        row_sum = 0
        for i in range(1, len(row)):
            row_sum += row.iloc[i]
        if len(row) - 1 == 0 or len(row) == 0:
            row_mean = 0
        else:
            row_mean = row_sum / (len(row) - 1)
        out_list.append(row_mean)
    return out_list

# EXECUTION

def main():
    # Create argument parser and parse arguments
    parser = argparse.ArgumentParser("Ranker is a module for assigning Students into Company Groups")
    parser.add_argument("--file", type=str, required=True, help="Input .csv File Path (Required)")
    parser.add_argument("--students", type=int, required=False, help="Number of students to process. If absent, will default to all students in --file (Optional)")
    parser.add_argument("--companies", type=int, default=30, help="Number of companies to include in algorithm. Must be less than or equal to those in the passed-in .csv file")
    parser.add_argument("--output_path", type=str, required=False, default='', help="Path to Output (Optional)")
    parser.add_argument("--suppress_terminal_output", type=bool, default=False, help="Option to suppress terminal output of program")
    args = parser.parse_args()

    input_file = args.file
    if input_file[-4:] != '.csv':
        return print(f"'{input_file}' argument for '--file' is invalid, must be a file with a .csv extension")

    # Make company column list
    company_columns = [0] + [x for x in range(1, args.companies + 1)]

    try:
    # Read input_file and num_students, and convert the file into our main dataframe
        if args.students:
            num_students = args.students
            main_df = pd.read_csv(input_file, usecols=company_columns, nrows=num_students)
        else:
            main_df = pd.read_csv(input_file, usecols=company_columns)
            num_students = main_df.shape[0]
    except (OSError, FileNotFoundError, pd.errors.ParserError) as e:
        return print(f"Error encountered. Program has terminated.\nError: {e}")

    # Here we replace all instances of NaN with 6 (FOR TESTING PURPOSES!!!)
    main_df.fillna(6, inplace=True)

    # Here we need to "trim" down the list into the top ten companies
    top_ten_companies = get_top_ten_companies(main_df)

    for column, data in main_df.items():
        if column == "Student":
            continue
        if column not in top_ten_companies:
            main_df = main_df.drop([column], axis=1)

    # Create the buckets (distribution) of students for each company
    company_buckets = calculate_company_buckets(num_students, 10)

    # Get the greatest number of slots from the first element of company_buckets for setting up 
    # the output csv columns
    max_slots = company_buckets[0]

    # Create an empty dataframe called group_df, with the following columns:
    # ['Company', 'CL2_norm_std, 'Student1 Name', 'Student1 Rank', ... , 'StudentN Name', 'StudentN Rank', 'Mean Rank']
    # Where N is equal to max_slots. We would like to keep a record of each student and their 
    # rank for each company. Note: In a dataframe, we are not required to have data in each column,
    # so in cases where later companies have N-1 Students, the empty columns will not effect the output.
    group_columns = ['Company', 'Des_Met', 'Desirability']
    for x in range(max_slots):
        group_columns.append(f'Student{x+1} Name')
        group_columns.append(f'Student{x+1} Rank')

    group_columns.append('Mean Rank')
    group_df = pd.DataFrame(columns=group_columns)

    # First, we should process the main dataframe and iteratively determine which company is the most
    # desirable, and then do group selection. After the selection, we should remove the students who were
    # grouped, and re-run the processing
    # We do this as many times as there are companies to assign students to
    for i in range(10):
        if not args.suppress_terminal_output:
            print(f"{Colors.OKBLUE}Round {i+1}{Colors.HEADER}")
            print('-----------------------------------------------------------------------------------------------------------------')

            # Select the most desirable
            highest_ranked = select_most_desirable_company(main_df, args.suppress_terminal_output)
            print('-----------------------------------------------------------------------------------------------------------------')
            print()
            print(f'Company Selected for Round {i+1}: {Colors.OKGREEN}{highest_ranked.name}{Colors.ENDC}')
            print(f'{Colors.HEADER}Statistics:{Colors.ENDC}\n{Colors.OKCYAN}{highest_ranked}{Colors.ENDC}')
            print()
        else:
            # Select the most desirable
            highest_ranked = select_most_desirable_company(main_df, args.suppress_terminal_output)

        company_name = highest_ranked.name
        
        # Create group_row_list for adding to group_df after selection
        group_row_list = [company_name, highest_ranked['Des_Met'], highest_ranked['Desirability']]

        # Make a new dataframe with the highest ranked company data, and the corresponding students
        hr_company_df = main_df[['Student', company_name]].copy()
        
        # Remove the highest-ranked companies column from the dataframe
        main_df.pop(company_name)

        # Here we can calculate each student's average ranks among what remains. This would encode their
        # "best chance" to be placed into their relative top-choice.
        remaining_rank_averages = get_remaining_rank_averages(main_df)

        # Add the remaining_rank_averages to hr_company_df for selection processing
        hr_company_df['remaining_rank_average'] = remaining_rank_averages

        # Select Student group for the highest-ranked company
        student_list = select_company_group(hr_company_df, company_buckets[i])
        
        # Add student names and rankings to the company group_row_list, and (optionally) 
        # print data to the terminal
        rank_sum = 0
        if not args.suppress_terminal_output:
            print(f'{Colors.HEADER}Students Selected for group:\n')
        for n in range(len(student_list[0])):
            group_row_list.append(student_list[0][n])
            group_row_list.append(student_list[1][n])
            rank_sum += student_list[1][n]
            if not args.suppress_terminal_output:
                print(f'{Colors.OKGREEN}{student_list[0][n]} - {student_list[1][n]}{Colors.ENDC}')
        if not args.suppress_terminal_output:
            print()
        
        # Calculate Mean Rank and add it to the group_row_list
        mean_rank = round(rank_sum / company_buckets[i], 4)

        if not args.suppress_terminal_output:
            print(f'{Colors.HEADER}Mean Group Rank - {Colors.OKCYAN}{mean_rank}{Colors.ENDC}')
            print()
        
        # We need to account for situations in which the number of students is less than the max group size
        # Therefore we need to look at the difference, and make sure to append None up to the position for
        # 'Mean Rank'
        size_diff = max_slots - company_buckets[i]
        for _ in range(size_diff):
            group_row_list.append(None)
            group_row_list.append(None)
        group_row_list.append(mean_rank)

        # Add group_row_list to group_df (by index)
        group_df.loc[i] = group_row_list

        # We now need to remove the selected students from the dataframe
        rows_to_remove = []
        for index, row in main_df.iterrows():
            if row['Student'] in student_list[0]:
                rows_to_remove.append(index)
        main_df = main_df.drop(rows_to_remove)

        # Reset the dataframe's index to ensure later iterations do not cause an error
        main_df = main_df.reset_index(drop=True)

    # Now that the main loop is finished, we need to convert our assembled group_df into a csv
    # and write it to an output path
    output_path = args.output_path + 'ranker_data.csv'
    group_df.to_csv(output_path, index=False)

    return


if __name__ == '__main__':
    main()