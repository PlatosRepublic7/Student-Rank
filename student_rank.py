# Sample script for student ranking algorithm
import argparse
from math import ceil
from faker import Faker  # For generating fake names
import random
import json


class Colors:
    # Text coloring class for use within Window's terminals
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


'''
Functions
'''
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


def create_company_ranks(companies: list, students: list) -> dict:
    '''
    Iterate through each company, and create a list of students who ranked into that choice.
    '''
    out_dict = {}

    # Constant for determining the size of groups based on how many students
    # there are compared to the number of companies
    #max_list_len = ceil(len(students) / len(companies))
    for i in range(len(companies)):
        #min_choice_students = ["" for x in range(len(max_list_len))]
        out_dict[companies[i]] = []
        for j in range(len(students)):
            for name, r_list in students[j].items():
                # Note that i corresponds to the company that the student has ranked within their
                # r_list
                out_dict[companies[i]].append({name: r_list[i]})

    return out_dict


def process_company_dict(company_dict:  dict, num_companies: int, num_students: int) -> dict:
    out_dict = {}
    cur_companies = num_companies

    # Determine the maximum number of students that can be assigned to a company group
    max_students = ceil(num_students / num_companies)

    while(cur_companies > 0):
        # First process each company to determine which one is the most desirable (lowest average)
        most_desirable = ""
        least_rank = 0.0
        for company, s_list in company_dict.items():
            rank_sum = 0
            for s_dict in s_list:
                for name, value in s_dict.items():
                    rank_sum += value
            company_average = rank_sum / len(s_list)
            if company_average > least_rank:
                most_desirable = company
                least_rank = company_average

        # Now that we have processed which is the most desirable, we pop it from the dictionary for
        # further processing
        out_company = {}
        out_company[most_desirable] = company_dict.pop(most_desirable)
        out_company['Average Rank'] = least_rank
        cur_companies -= 1
        
        # We will now iterate through the selected company and the students who ranked it.
        # We will select a random starting index, and add students to the company_group using the 
        # following rule: If the student's rank is STRICTLY LESS than the max rank within the group,
        # remove the student with the max rank, and replace them with the current student.
        # We then add the students name to ignore_names_list, where they will be popped from all
        # company dictionaries during subsequent processing.
        company_group = [{f"NULL{i}": 11} for i in range(max_students)]
        max_rank = 11
        starting_index = random.randint(0, len(out_company[most_desirable]) - 1)
               
        for j in range(len(out_company[most_desirable])):
            index = (starting_index + j) % len(out_company[most_desirable])
            for s_name, s_rank in out_company[most_desirable][index].items():
                if s_rank < max_rank:
                    found_max = False
                    for k in range(len(company_group)):
                        if found_max:
                            break
                        for student_name in company_group[k].keys():
                            if company_group[k][student_name] == max_rank:
                                company_group.pop(k)
                                company_group.append({s_name: s_rank})
                                max_rank = get_max_value(company_group)
                                found_max = True
                                break
                        
        ignore_names_list = []
        for c_student in company_group:
            for c_name in c_student.keys():
                ignore_names_list.append(c_name)
        
        remove_names_from_companies(ignore_names_list, company_dict)
        out_dict[most_desirable] = company_group

    return out_dict


def remove_names_from_companies(ignore_names_list: list, company_dict: dict) -> None:
    for company, c_list in company_dict.items():
        ignore_name_indexes = []
        for i in range(len(c_list)):
            for name in c_list[i].keys():
                if name in ignore_names_list:
                    ignore_name_indexes.append(i)
        num_removed = 0
        for index in ignore_name_indexes:
            c_list.pop(index - num_removed)
            num_removed += 1


def get_max_value(in_list: list) -> int:
    max = 0
    for i in range(len(in_list)):
        for key, value in in_list[i].items():
            if value > max:
                max = value
    return max


'''
Data Structures - Represents what kind of format we would want the companies or student names and
rankings to look like.
'''
# Ten example "companies"
companies = [
    "TechNova",
    "GreenFields",
    "Skyline Dynamics",
    "Blue Horizon",
    "Pioneer Solutions",
    "BrightPath",
    "NextEra Innovations",
    "Quantum Leap Systems",
    "EcoSphere Enterprises",
    "Visionary Ventures"
]

'''
Program Entry Point
'''

def main():
    # Create argument parser and parse arguments
    parser = argparse.ArgumentParser(description="Sample Program for Producing Student Ranking of Companies")
    parser.add_argument("--students", type=int, default=10)
    parser.add_argument("--file", type=str, help="Path to Input File")
    args = parser.parse_args()

    file_path = args.file


    company_dict = {}
    student_list = []

    if file_path != "":
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            companies = data['Companies']
            for n in range(args.students):
                student_list.append(data['Students'][n])
            company_dict = create_company_ranks(companies, student_list)
    else:
        # Generate a list of student ranks
        student_list = gen_random_student_ranking(args.students)

        # Generate company list with corresponding student names and rankings
        company_dict = create_company_ranks(companies, student_list)
    

    print(f"{Colors.HEADER}Companies{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{companies[0]} | {companies[1]} | {companies[2]} | {companies[3]} | {companies[4]} | {companies[5]} | {companies[6]} | {companies[7]} | {companies[8]} | {companies[9]}{Colors.ENDC}\n")

    print(f"{Colors.HEADER}Generated List of Students with a list of their company rankings{Colors.ENDC}")
    for i in range(len(student_list)):
        print(student_list[i])
    print()

    print(f"{Colors.HEADER}Company list with corresponding students and their rankings{Colors.ENDC}")
    for key, value in company_dict.items():
        print(f'{Colors.OKGREEN}{key}{Colors.ENDC} | {value}\n')

    # Find the most desirable company by lowest average rank
    company_groups = process_company_dict(company_dict, len(companies), len(student_list))

    # Compute Average Rank for each assigned group
    average_desire_per_group = {}
    for company, group_list in company_groups.items():
        total_rank = 0
        for m in range(len(group_list)):
            for value in group_list[m].values():
                total_rank += value
        group_average = total_rank / len(group_list)
        average_desire_per_group[company] = round(group_average, 2)
    # Compute average desirability for all the groups
    g_sum = 0
    for g_average in average_desire_per_group.values():
        g_sum += g_average
    t_average = g_sum / len(average_desire_per_group)

    print()
    print(f"{Colors.HEADER}Company Groups{Colors.ENDC}")
    for key, value in company_groups.items():
        print(f'{Colors.OKGREEN}{key}{Colors.ENDC} | {value}')
    print()

    print(f"{Colors.HEADER}Average Desirability Per Company (After Assignment){Colors.ENDC}")
    for key, value in average_desire_per_group.items():
        print(f'{Colors.OKGREEN}{key}{Colors.ENDC} | {Colors.OKCYAN}{value}{Colors.ENDC}')
    print()

    print(f'{Colors.HEADER}Average Desirability of All Companies:{Colors.ENDC} {Colors.OKCYAN}{round(t_average, 2)}{Colors.ENDC}')


if __name__ == "__main__":
    main()