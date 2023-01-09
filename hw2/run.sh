python3 run_context.py \--context_file "${1}" \--test_file "${2}" \--output_dir ./ffffff/  
python3 run_question.py \--validation_file result_context.csv \--save_file "${3}" \--output_dir ./final/
rm context.zip
rm question_answering.zip
rm -rf ffffff
rm -rf final
rm -rf result_context.csv
rm test.csv