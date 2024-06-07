from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import confidence_estimation_utils
import data_reading_utils
from transformers import HfArgumentParser
from cmd_line_arguments import ScriptArguments
import logging
import pickle
import os
from transformers import set_seed
from tqdm import tqdm
import text2sql_dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger = logging.getLogger(__name__)

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        tokenizer.convert_tokens_to_ids(";"),
        tokenizer.convert_tokens_to_ids("';"),
        tokenizer.convert_tokens_to_ids('";')
        # tokenizer.convert_tokens_to_ids("#"),
        # tokenizer.convert_tokens_to_ids(";"),
        # tokenizer.convert_tokens_to_ids("\n\n")
    ]

    if args.use_lora:
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_name,
                                                         device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map='cuda')

    model.generation_config.pad_token_ids = tokenizer.pad_token_id

    # read test data

    testing_sft_dataset = []
    if args.sql_dataset_name == "pauq":
        testing_sft_dataset = data_reading_utils.create_pauq_sft_dataset_v2(args.path_to_testing_file,
                                                                            args.tables_info_path,
                                                                            tokenizer,
                                                                             phase="test",
                                                                             try_one_batch=args.try_one_batch)
    elif args.sql_dataset_name == "ehrsql":
        testing_sft_dataset = data_reading_utils.create_ehrsql_sft_dataset(args.path_to_testing_file,
                                                                           args.tables_info_path,
                                                                           phase="test",
                                                                           try_one_batch=args.try_one_batch)
    testing_sft_dataset = text2sql_dataset.Text2SQLDataset(testing_sft_dataset, tokenizer, args.max_seq_length, device)
    print(f'Total testing samples = {len(testing_sft_dataset)}')

    eval_dataloader = DataLoader(testing_sft_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size)

    ids_list, prediction_list, scores_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            sample_id = batch['id']
            input_length = batch['input_ids'].shape[1]
            outputs = model.generate(input_ids=batch['input_ids'],
                                    max_new_tokens=args.max_new_tokens,
                                    num_beams=args.num_beams,
                                    eos_token_id=terminators,
                                    output_logits=True,
                                    return_dict_in_generate=True,
                                    pad_token_id=tokenizer.eos_token_id)

            generated_sequences = outputs["sequences"].cpu() if "cuda" in device else outputs["sequences"]

            entropy_scores = confidence_estimation_utils.maximum_entropy_confidence_score_method(generation_scores=outputs["logits"],
                                                                                                device=device)
            entropy_scores = confidence_estimation_utils.truncate_scores(generated_sequences=generated_sequences,
                                                            scores=entropy_scores,
                                                            tokenizer=tokenizer)
            max_entropy_scores = [max(score_list) for score_list in entropy_scores]
            scores_list += max_entropy_scores
            decoded_preds = tokenizer.batch_decode(generated_sequences[:, input_length:], skip_special_tokens=True)
            predictions = [data_reading_utils.generated_query_simple_processor(pred) for pred in decoded_preds]
            prediction_list += predictions
            ids_list += sample_id

    print('Inference completed!')

    result_dict = dict()
    for id_, pred_sql, score in zip(ids_list, prediction_list, scores_list):
        result_dict[id_] = {
            "sql": pred_sql,
            "score": score
        }


    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(args.path_to_testing_file).split('.')[0]
    if args.try_one_batch:
        filename = f"{filename}_one_batch_inference_result.pkl"
    else:
        filename = f"{filename}_inference_result.pkl"
    save_path = os.path.join(output_dir, filename)
    logger.info("Writing model predictions to json file...")

    pickle.dump(result_dict, open(save_path, 'wb'))

    if args.try_one_batch == False:
        filename = filename.split('.')[0]
        filename = f"{filename}_sql_predictions.txt"
        save_path = os.path.join(output_dir, filename)
        with open(save_path, 'w') as f:
            for sql in prediction_list:
                f.write(f"{sql}\n")




    # model_id = "/home/etutubalina/somov-od/llama3/Meta-Llama-3-8B-Instruct/hf_converted"
    #
    # pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
    #                                  device_map="auto")
    # print(pipeline("Hey how are you doing today? Answer in Russian"))

    # # Example of using the fine-tuned model
    # messages = [
    #     {"role": "system", "content": "You are a helpful medical chatbot."},
    #     {"role": "user", "content": "I have a headache and fever."},
    # ]
    #
    # input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
    #     model.device)
    # outputs = model.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7)
    # response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    # print(response)


