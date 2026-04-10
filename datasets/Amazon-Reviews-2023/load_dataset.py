from datasets import load_dataset


amazon = load_dataset('./Amazon-Reviews-2023.py', 'raw_meta_All_Beauty', split='test')
amazon.save_to_disk('.')