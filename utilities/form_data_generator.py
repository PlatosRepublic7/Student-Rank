import csv
import random
import argparse


def get_file_data(companies_path, names_path) -> list:
    # The format for funtion return data is [NAME, COMPANY, EMAIL]
    companies_data = []
    with open(companies_path, mode='r') as c_file:
        company_names = c_file.readlines()
        for company_name in company_names:
            company_name = company_name.rstrip(' \n')
            companies_data.append(company_name)
    c_file.close()

    name_data = []
    with open(names_path, mode='r') as n_file:
        names_list = n_file.readlines()
        for names_line in names_list:
            names_line_list = names_line.split('-')
            names_line_list[0] = names_line_list[0].rstrip(' \n')
            names_line_list[1] = names_line_list[1].rstrip(' \n').lstrip()
            name_data.append(names_line_list)
    n_file.close()
    return [companies_data, name_data]


def create_company_entries(file_data) -> list:
    companies = []
    for i in range(len(file_data[0])):
        c_name = file_data[0][i]
        email = file_data[1][i][1]
        company_entry = c_name + f' - {email}'
        companies.append(company_entry)

    return companies


def create_data_entries(company_entries, names) -> list:
    out_data = []
    for i in range(len(names)):
        entry_dict = {'Timestamp': 0, 'Email': names[i][1], 'Name': names[i][0]}
        #entry_dict = {'Id': i, 'Start time': 0, 'Completion time': 0, 'Email': names[i][1], 'Name': names[i][0], 'Description': ''}
        for j in range(len(company_entries)):
            entry_dict[company_entries[j]] = None

        random_companies = random.sample(company_entries, 5)
        # for j in range(len(random_companies)):
        #     if j == len(random_companies) - 1:
        #         entry_dict['Description'] += random_companies[j]
        #     else:
        #         entry_dict['Description'] += random_companies[j] + ';'
        for k in range(len(random_companies)):
            entry_dict[random_companies[k]] = k + 1
        out_data.append(entry_dict)

    return out_data


def get_extra_names(names_path, num_names):
    name_data = []
    with open(names_path, mode='r') as n_file:
        names_list = n_file.readlines()
        for i in range(num_names):
            names_line_list = names_list[i].split('-')
            names_line_list[0] = names_line_list[0].rstrip(' \n')
            names_line_list[1] = names_line_list[1].rstrip(' \n').lstrip()
            name_data.append(names_line_list)
    n_file.close()
    return name_data


def main():
    parser = argparse.ArgumentParser("Generate csv data for ranker2.py")
    parser.add_argument('--companies', type=str, default='./utilities/companies.txt')
    parser.add_argument('--students', type=str, default='./utilities/names_and_emails.txt')
    parser.add_argument('--extra', type=int, default=0, help="Number of students to add to data (without their own companies)")
    args = parser.parse_args()
    file_data = get_file_data(args.companies, args.students)
    # We need to create the columns of the data table with each company as a column, with a '-' separating
    # the company name and the email of the person who proposed it
    company_entries = create_company_entries(file_data)
    data_columns = ['Timestamp', 'Email', 'Name'] + company_entries
    names = file_data[1]
    if args.extra > 0:
        names_to_add = args.extra
        extra_names = get_extra_names('./extra_names_and_emails.txt', names_to_add)
        for j in range(len(extra_names)):
            names.append(extra_names[j])

    data = create_data_entries(company_entries, names)

    with open('form_test_data_cc.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_columns)

        writer.writeheader()
        writer.writerows(data)

    return


if __name__ == '__main__':
    main()