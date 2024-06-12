import pandas as pd

def find_foreign_keys_MYSQL_like(db_name, foreign_keys_df):
    df = foreign_keys_df[foreign_keys_df['Database name'] == db_name]
    output = "["
    for index, row in df.iterrows():
        output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
    output= output[:-1]
    if len(output) == 0:
        output = "[]\n"
    else:
        output += "]"
    return output


def find_fields_MYSQL_like(db_name, schema_df):
    df = schema_df[schema_df['Database name'] == db_name]
    df = df.groupby(' Table Name')
    output = ""
    for name, group in df:
        output += "Table " +name+ ', columns = ['
        for index, row in group.iterrows():
            output += row[" Field Name"]+','
        output = output[:-1]
        output += "]\n"
    return output


def find_primary_keys_MYSQL_like(db_name, primary_keys_df):
    df = primary_keys_df[primary_keys_df['Database name'] == db_name]
    output = "["
    for index, row in df.iterrows():
        output += row['Table Name'] + '.' + row['Primary Key'] +','
    output = output[:-1]
    if len(output) == 0:
        output = "[]\n"
    else:
        output += "]\n"
    return output

def create_schema_string(db_name, schema_df, foreign_keys_df, primary_keys_df):
    schema_string = find_fields_MYSQL_like(db_name, schema_df)
    schema_string += "Foreign_keys = " + find_foreign_keys_MYSQL_like(db_name, foreign_keys_df) + '\n'
    schema_string += "Primary_keys = " + find_primary_keys_MYSQL_like(db_name, primary_keys_df)
    return schema_string

def create_schema_df(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    schema_df = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    primary_df = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    foreign_df = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return schema_df,primary_df,foreign_df

