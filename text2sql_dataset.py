from torch.utils.data import Dataset


class Text2SQLDataset(Dataset):
    def __init__(self, sft_dataset, tokenizer, max_length, device):

        self.prompts_ids = [sample['id'] for sample in sft_dataset]
        prompts_list = [sample['text'] for sample in sft_dataset]
        self.tokenized_prompts = tokenizer(prompts_list, max_length=max_length,
                                           truncation=True, padding=True, add_special_tokens=False,
                                           return_tensors='pt')
        self.device = device

    def __len__(self):
        return len(self.prompts_ids)

    def __getitem__(self, idx):
        sample_id = self.prompts_ids[idx]
        input_ids, attention_mask = self.tokenized_prompts['input_ids'][idx].to(self.device), \
                                    self.tokenized_prompts['attention_mask'][idx].to(self.device)

        return {"id": sample_id, "input_ids": input_ids, "attention_mask": attention_mask}