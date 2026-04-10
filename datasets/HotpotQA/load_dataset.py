from datasets import load_dataset

hotpot_qa = load_dataset('./hotpot_qa.py','fullwiki')
hotpot_qa.save_to_disk('.')