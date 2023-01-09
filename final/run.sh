
gdown https://drive.google.com/uc?id=1LDh_o1jlLFq-KLW7jYRFnDSFFJdNzLnS
unzip model_distill.zip -d ./
python simulator.py --split test --model_name_or_path no_more_game_read --output output_reproduce.jsonl


# https://drive.google.com/file/d/1LDh_o1jlLFq-KLW7jYRFnDSFFJdNzLnS/view?usp=sharing