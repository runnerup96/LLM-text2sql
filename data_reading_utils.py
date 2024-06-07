import json
from datasets import Dataset
import os


INSTRUCTION = "You are an text to SQL query translator. " \
              "Users will ask you questions in English and you will generate only a SQL query as assistant answer" \
              " based on the provided SCHEMA and DATABASE NAME.\n" \
              "{database_name}\n" \
              "{schema}"
RESPONSE_TEMPLATE = "SQL:"


INSTRUCTION_V2 = "You are text-to-SQL assistant. " \
                 "You get question from user along with database schema. " \
                 "You have to generate a correct SQL query as response. " \
                 "Use only SQL light syntax. " \
                 "Use only entities from database schema in SQL query. "
RESPONSE_TEMPLATE_V2 = "```sql"

def form_database_dict(db_sample):
    table_list = db_sample['table_names_original']
    column_list = db_sample['column_names_original']

    table2column_dict = dict()
    for column_info in column_list[1:]:
        table_idx, column_name = column_info
        table_name = table_list[table_idx]
        if table_name not in table2column_dict:
            table2column_dict[table_name] = [column_name]
        else:
            table2column_dict[table_name].append(column_name)

    return table2column_dict


def create_pauq_sft_dataset_for_defog(data_path, tables_info_path, phase='train', try_one_batch=False, batch_size=4):
    data = json.load(open(data_path, 'r'))
    tables_info = json.load(open(tables_info_path, 'r'))

    db2info_dict = dict()
    for sample in tables_info:
        db_name = sample['db_id']
        db_info = form_database_dict(sample)
        db2info_dict[db_name] = db_info

    sft_dataset_list = []
    for sample in data:
        sample_id = sample["id"]
        db_id = sample["db_id"]
        input_text = sample["question"]
        schema_info_dict = db2info_dict[db_id]

        DEFOG_RESPONSE_TEMPLATE='```sql'
        DEFOG_INSTRUCTION = "Generate a SQLight query based on SCHEMA and DATABASE NAME\n" \
                              "{database_name}\n" \
                              "{schema}"

        database_name_str = f"# DATABASE NAME: {db_id}"
        schema_str = "\n".join(
            [f"# {table}({', '.join(attributes)})" for table, attributes in schema_info_dict.items()])
        schema_str = f"# SCHEMA:\n{schema_str}"
        instruction = DEFOG_INSTRUCTION.format(database_name=database_name_str,
                                         schema=schema_str)


        if phase == 'train':
            out_text = f'{DEFOG_RESPONSE_TEMPLATE}{sample["query"]}'
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Generate a SQL query to answer this question: `{input_text}`\n"
                f"{instruction}"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"The following SQL query best answers the question `{input_text}`:\n"
                f"{str(out_text)}"
                f"<|eot_id|>"
            )
        else:
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Generate a SQL query to answer this question: `{input_text}`\n"
                f"{instruction}"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"The following SQL query best answers the question `{input_text}`:\n"
                f"{DEFOG_RESPONSE_TEMPLATE}"
            )
        formatted_prompt = "".join(formatted_prompt)
        sft_dataset_list.append({'id': sample_id, 'text': formatted_prompt})
    if try_one_batch:
        sft_dataset_list = sft_dataset_list[-batch_size:]
    sft_dataset = Dataset.from_list(sft_dataset_list)
    return sft_dataset

    pass


def create_pauq_sft_dataset_v2(data_path, tables_info_path, tokenizer, phase='train', try_one_batch=False, batch_size=4):
    data = json.load(open(data_path, 'r'))
    tables_info = json.load(open(tables_info_path, 'r'))

    db2info_dict = dict()
    for sample in tables_info:
        db_name = sample['db_id']
        db_info = form_database_dict(sample)
        db2info_dict[db_name] = db_info

    sft_dataset_list = []
    for sample in data:
        sample_id = sample["id"]
        db_id = sample["db_id"]
        input_text = sample["question"]
        sql = sample["query"]
        schema_info_dict = db2info_dict[db_id]
        schema_str = "\n".join([f"{table}({', '.join(attributes)})" for table, attributes in schema_info_dict.items()])

        user_task = (f"Generate a SQL query to answer this question: `{input_text}`\n"
                     f"Database schema: {schema_str} \n")

        if phase == 'train':

            assistant_answer = (f"{RESPONSE_TEMPLATE_V2}{sql}")
            chat = [
                {"role": "system", "content": INSTRUCTION_V2},
                {"role": "user", "content": user_task},
                {"role": "assistant", "content": assistant_answer}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        else:
            chat = [
                {"role": "system", "content": INSTRUCTION_V2},
                {"role": "user", "content": user_task}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            formatted_prompt += RESPONSE_TEMPLATE_V2

        sft_dataset_list.append({'id': sample_id, 'text': formatted_prompt})
    if try_one_batch:
        sft_dataset_list = sft_dataset_list[-batch_size:]
    sft_dataset = Dataset.from_list(sft_dataset_list)
    return sft_dataset


def create_pauq_sft_dataset(data_path, tables_info_path, phase='train', try_one_batch=False, batch_size=4):
    data = json.load(open(data_path, 'r'))
    tables_info = json.load(open(tables_info_path, 'r'))

    db2info_dict = dict()
    for sample in tables_info:
        db_name = sample['db_id']
        db_info = form_database_dict(sample)
        db2info_dict[db_name] = db_info

    sft_dataset_list = []
    for sample in data:
        sample_id = sample["id"]
        db_id = sample["db_id"]
        input_text = sample["question"]
        schema_info_dict = db2info_dict[db_id]
        database_name_str = f"# DATABASE NAME: {db_id}"
        schema_str = "\n".join([f"# {table}({', '.join(attributes)})" for table, attributes in schema_info_dict.items()])
        schema_str = f"# SCHEMA:\n{schema_str}"
        instruction = INSTRUCTION.format(database_name=database_name_str,
                                         schema=schema_str)
        if phase == 'train':
            out_text = f'{RESPONSE_TEMPLATE}{sample["query"]}'
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{instruction}<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{str(out_text)}<|eot_id|>"
            )
        else:
            formatted_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{instruction}<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{RESPONSE_TEMPLATE}"
            )
        formatted_prompt = "".join(formatted_prompt)
        sft_dataset_list.append({'id': sample_id, 'text': formatted_prompt})
    if try_one_batch:
        sft_dataset_list = sft_dataset_list[-batch_size:]
    sft_dataset = Dataset.from_list(sft_dataset_list)
    return sft_dataset

def create_ehrsql_sft_dataset(dataset_folder_path, tables_info_path, phase='train', try_one_batch=False, batch_size=4):
    inputs_path = os.path.join(dataset_folder_path, "data.json")
    targets_path = os.path.join(dataset_folder_path, "label.json")
    inputs = json.load(open(inputs_path, 'r'))
    inputs = inputs['data']
    inputs = {sample['id']:sample['question'] for sample in inputs}

    table_info = json.load(open(tables_info_path, 'r'))
    mimic_table_info = table_info[0]
    mimic_table_info = form_database_dict(mimic_table_info)
    db_id = "mimic_iv"
    database_name_str = f"# DATABASE NAME: {db_id}"
    mimic_table_str = "\n".join([f"# {table}({', '.join(attributes)})" for table, attributes in mimic_table_info.items()])
    schema_str = f"# SCHEMA:\n{mimic_table_str}"


    targets = json.load(open(targets_path, 'r'))
    sft_dataset_list = []
    for key in inputs:
        input_text = inputs[key]
        out_text = f"{RESPONSE_TEMPLATE}{targets[key]}"
        instruction = INSTRUCTION.format(database_name=database_name_str,
                       schema=schema_str)
        if out_text != 'null':
            if phase == 'train':
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{instruction}<|eot_id|>\n"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{input_text}<|eot_id|>\n"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"{str(out_text)}<|eot_id|>"
                )
            else:
                formatted_prompt = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{instruction}<|eot_id|>\n"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{input_text}<|eot_id|>\n"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"{RESPONSE_TEMPLATE}"
                )
            formatted_prompt = "".join(formatted_prompt)
            sft_dataset_list.append({'id': key, 'text': formatted_prompt})

    if try_one_batch:
        sft_dataset_list = sft_dataset_list[-batch_size:]
    return sft_dataset_list

def generated_query_simple_processor(string):
    string = string.replace('\n', '')
    if ';' in string:
        string = string.split(';')
        result = string[0]
    else:
        result = string
    return result


if __name__ == "__main__":
    from transformers import AutoTokenizer

    pauq_train_path = "/home/somov/text2sql_llama_3/data/pauq/pauq_xsp_train.json"
    pauq_test_path = "/home/somov/text2sql_llama_3/data/pauq/pauq_xsp_test.json"
    tables_path = "/home/somov/text2sql_llama_3/data/pauq/tables.json"

    model_name = "/home/somov/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    sft_dataset = create_pauq_sft_dataset_v2(pauq_train_path, tables_path, tokenizer, phase='train')
    print()

    # result_list = []
    # for idx in range(len(sft_dataset)):
    #     result_list.append(sft_dataset[idx]['text'])

    # json.dump(result_list, open("/Users/somov-od/Documents/phd/projects/text2sql_llama_3/data/pauq/train_sft.json", 'w'),
    #           ensure_ascii=False, indent=4)


    # ehrsql_path = "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/data/ehrsql/train"
    # ehrsql_tables_path = "/Users/somov-od/Documents/phd/projects/text2sql_llama_3/data/ehrsql/tables.json"
    # sft_dataset = create_ehrsql_sft_dataset(ehrsql_path, ehrsql_tables_path, phase="test")


