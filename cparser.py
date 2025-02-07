import pandas as pd
import numpy as np
import csv


class CParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.companies = {}
        self.students = []
        self.proposed_companies = {}
        self.df_ready_columns = ['Student']
        self.df_ready_data = []

    
    def _parse_file(self):
        csv_data = []
        try:
            with open(self.file_path, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    csv_data.append(row)
            file.close()
        except Exception as e:
            return f'Parsing Error:\n{e}'

        for entry in csv_data:
            student_dict = {entry['Name']: entry['Email'], 'Propsed Company': '', 'Rankings': {}}
            # skip_keys = ['Id', 'Start time', 'Completion time', 'Email', 'Name']
            # for key, value in entry.items():
            #     if key not in skip_keys:
            #         rank_string_list = value.split(';')
            #         rank = 1
            #         for company_string in rank_string_list:
            #             company_string_list = company_string.split('-')
            #             company_name = company_string_list[0].strip(' ').replace('\xa0', '').lower().capitalize()
            #             start = company_string_list[-1].find("(") + 1
            #             end = company_string_list[-1].find(")")
            #             company_proposer = company_string_list[-1][start:end]
            #             if company_proposer == entry['Email']:
            #                 student_dict['Propsed Company'] = company_name
            #             if company_name not in self.companies.keys():
            #                 self.companies[company_name] = company_proposer
            #             student_dict['Rankings'][company_name] = rank
            #             if rank < 6:
            #                 rank += 1
            #         self.students.append(student_dict)
            skip_keys = ['Timestamp', 'Email', 'Name']
            for key, value in entry.items():
                if key not in skip_keys:
                    key_list = key.split('-')
                    company_name = key_list[0].strip(' ')
                    company_proposer = key_list[1].strip(' ')
                    if company_proposer == entry['Email']:
                        student_dict['Propsed Company'] = company_name
                    if company_name not in self.companies.keys():
                        self.companies[company_name] = company_proposer
                    if value == '':
                        rank_value = 6
                    else:
                        rank_value = int(value)
                    student_dict['Rankings'][company_name] = rank_value
            self.students.append(student_dict)


    def make_data_frame(self, num_students=0) -> pd.DataFrame:
        self._parse_file()
        self.df_ready_columns = ['Student']
        self.df_ready_data = []
        self.proposed_companies = {}

        for company in self.companies.keys():
            self.df_ready_columns.append(company)

        for i in range(len(self.students)):
            zipped = zip(self.students[i].keys(), self.students[i].values())
            zlist = list(zipped)
            row_data = [zlist[0][0]]
            for company in self.df_ready_columns:
                if company == 'Student':
                    continue
                row_data.append(self.students[i]['Rankings'][company])
            self.df_ready_data.append(row_data)
            self.proposed_companies[zlist[0][0]] = zlist[1][1]
        
        df = pd.DataFrame(columns=self.df_ready_columns)
        if num_students == 0:
            num_students = len(self.df_ready_data)
        for i in range(num_students):
            df.loc[i] = self.df_ready_data[i]

        return df


if __name__ == '__main__':
    cparser = CParser('./data/form_test_data.csv')
    mdf = cparser.make_data_frame(30)
        