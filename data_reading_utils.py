import json
from datasets import Dataset
import os
import create_db_schema_string
from tqdm import tqdm
import preprocessing_utils

INSTRUCTION = """
You are text-to-SQL assistant. You get QUESTION and DATABASE SCHEMA from user.  
Generate only a correct SQL query in response:
"""


def create_ehrsql_sft_dataset(data_path, tables_info_path, tokenizer, phase='train', try_one_batch=False, batch_size=4):

    question_path = os.path.join(data_path, "data.json")
    question_list = json.load(open(question_path, 'r'))
    if "data" in question_list:
        question_list = question_list["data"]

    label_path = os.path.join(data_path, "label.json")
    label_dict = json.load(open(label_path, 'r'))

    ehrsql_schema, ehrsql_primary, ehrsql_foreign = create_db_schema_string.create_schema_df(tables_info_path)

    db_name = "mimic_iv"
    schema_str = create_db_schema_string.create_schema_string(db_name, ehrsql_schema, ehrsql_foreign, ehrsql_primary)

    sft_dataset_list = []
    for sample in tqdm(question_list):
        sample_id = sample["id"]
        input_text = preprocessing_utils.process_input_question(sample["question"])
        sql = label_dict[sample_id]

        user_task = (f"QUESTION: {input_text}\n"
                     f"DATABASE SCHEMA: {schema_str}\n")

        if phase == 'train':
            if sql != 'null':
                assistant_answer = preprocessing_utils.normalize_sql_query(sql)
                chat = [
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": user_task},
                    {"role": "assistant", "content": assistant_answer}
                ]
                formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            else:
                continue
        else:
            chat = [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": user_task}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        sft_dataset_list.append({'id': sample_id, 'text': formatted_prompt})
    if try_one_batch:
        sft_dataset_list = sft_dataset_list[:batch_size]
    sft_dataset = Dataset.from_list(sft_dataset_list)
    return sft_dataset


def create_pauq_sft_dataset(data_path, tables_info_path, tokenizer, phase='train', try_one_batch=False, batch_size=4):
    data = json.load(open(data_path, 'r'))
    tables_info = json.load(open(tables_info_path, 'r'))

    spider_schema, spider_primary, spider_foreign = create_db_schema_string.create_schema_df(tables_info_path)

    db2schema_str_dict = dict()
    for sample in tables_info:
        db_name = sample['db_id']
        db_string = create_db_schema_string.create_schema_string(db_name, spider_schema, spider_foreign, spider_primary)
        db2schema_str_dict[db_name] = db_string

    sft_dataset_list = []
    for sample in tqdm(data):
        sample_id = sample["id"]
        db_id = sample["db_id"]
        input_text = preprocessing_utils.process_input_question(sample["question"])
        sql = sample["query"]
        schema_str = db2schema_str_dict[db_id]

        user_task = (f"QUESTION: {input_text}\n"
                     f"DATABASE SCHEMA: {schema_str}\n")

        if phase == 'train':

            assistant_answer = preprocessing_utils.normalize_sql_query(sql)
            chat = [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": user_task},
                {"role": "assistant", "content": assistant_answer}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        else:
            chat = [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": user_task}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        sft_dataset_list.append({'id': sample_id, 'text': formatted_prompt})
    if try_one_batch:
        sft_dataset_list = sft_dataset_list[:batch_size]
    sft_dataset = Dataset.from_list(sft_dataset_list)
    return sft_dataset



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


    ehrsql_train_path = "/home/somov/text2sql_llama_3/data/ehrsql/train"
    ehrsql_test_path = "/home/somov/text2sql_llama_3/data/ehrsql/test"
    tables_path = "/home/somov/text2sql_llama_3/data/ehrsql/tables.json"


    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token="")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
    MAX_LENGTH = 1024
    # sft_dataset = create_pauq_sft_dataset(pauq_test_path, tables_path, tokenizer, phase='train')
    sft_dataset = create_ehrsql_sft_dataset(ehrsql_test_path, tables_path, tokenizer, phase='test')

    # TS_3630 and so on - big schema(more than 1024)
    number_of_max_length, avg_prompt_length = 0, 0
    for sample in tqdm(sft_dataset):
        id_, text = sample['id'], sample['text']
        encoded_text = tokenizer(text, truncation=True, padding=True, max_length=MAX_LENGTH, add_special_tokens=False)
        decoded_text = tokenizer.decode(encoded_text['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        if len(encoded_text['input_ids']) == MAX_LENGTH:
            number_of_max_length += 1
            continue
        else:
            assert text == decoded_text

        avg_prompt_length += len(encoded_text['input_ids'])
    print('Avg prompt length: ', round(avg_prompt_length / len(sft_dataset), 2))
    print('Number of max length samples: ', number_of_max_length)



