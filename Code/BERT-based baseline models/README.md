
#### Packages installation
```python
!pip install git+https://github.com/huggingface/transformers
!pip install -r './requirements.txt'
# os.environ["WANDB_DISABLED"] = "true" # if you are running the code in jupyter notebook (else comment this line)
```

#### Training the models
```python
!python './run_classification.py' \
    --model_name_or_path  microsoft/mdeberta-v3-base \   # replace it with the model of your choice
    --train_file "../Dataset/TeClass_training_data.csv" \
    --validation_file "../Dataset/TeClass_development_data.csv" \
    --test_file "../Dataset/TeClass_testing_data.csv" \
    --shuffle_train_dataset \
    --metric_name "accuracy" \
    --text_column_name "headline,article" \
    --text_column_delimiter "[SEP]" \
    --label_column_name label \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --output_dir './model_checkpoints/' \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --save_safetensors False \
    --overwrite_output_dir True | tee logs.txt

```

#### Inference
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

# Path to the directory containing your saved model checkpoints
checkpoint_dir = "./model_checkpoints/best_model_checkpoint" # replace it with your best model checkpoint path

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

config = AutoConfig.from_pretrained(checkpoint_dir)
# Get the list of class labels from the configuration
class_labels = list(config.id2label.values())
print(class_labels)

def hc_demo(headline, article):
    # Concatenate inputs with [SEP] token
    input_text = f"{headline}[SEP]{article}"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class index
    predicted_class_index = torch.argmax(outputs.logits, dim=1).item()

    # Map predicted class index to label
    predicted_class_label = class_labels[predicted_class_index]
    
    predicted_class_label_map={'HREL':'Highly Related', 'MREL':'Moderately Related', 
                               'LREL':'Least Related'}
    #print(predicted_class_label)
    return predicted_class_label_map[predicted_class_label]
headline = "" # enter the headline text here
article  = "" # enter the article text here
predicted_class = hc_demo(headline, article)
print(predicted_class)
```
