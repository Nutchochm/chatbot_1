import pandas as pd
import torch
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, PreTrainedTokenizerFast
from peft import get_peft_model, LoraConfig 
from datasets import load_dataset, Value, Features, DatasetDict
from sklearn.model_selection import train_test_split


def excel_csv(excel):    
    try:
        # Determine if input file is Excel or CSV
        if excel.endswith('.csv'):
            # Load CSV into a DataFrame
            data = pd.read_csv(excel)
            print("Loaded CSV data:", data)
        else:
            # Load Excel file into a DataFrame
            data = pd.read_excel(excel)
            print("Loaded Excel data:", data)

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(data)

        # Convert 'Reference' column to string
        def convert_reference_to_string(examples):
            examples['Reference'] = [str(val) for val in examples['Reference']]
            return examples
        
        # Apply the conversion to 'Reference' column
        dataset = dataset.map(convert_reference_to_string, batched=True)

        # Define features explicitly (if needed)
        features = Features({
            'Question': Value('string'),
            'Input': Value('string'),
            'Answer': Value('string'),
            'Reference': Value('string')  # Convert Reference to string
        })

        # Cast dataset features
        dataset = dataset.cast(features)

        print("Processed dataset:", dataset)

        # Define CSV file path
        base_name = os.path.splitext(os.path.basename(excel))[0]  # Get file name without extension
        output_dir = os.path.dirname(excel)  # Use the same directory as input file

        # Check if directory is writable
        if not os.access(output_dir, os.W_OK):
            print(f"Directory not writable. Defaulting to /tmp.")
            output_dir = "/tmp" if os.name != 'nt' else "C:\\Temp"  # Use /tmp for Linux/Mac, C:\Temp for Windows
            os.makedirs(output_dir, exist_ok=True)  # Ensure the temp directory exists

        csv_name = os.path.join(output_dir, f"{base_name}.csv")  # Final CSV path

        # Save dataset as CSV
        dataset_df = dataset.to_pandas()  # Convert Hugging Face Dataset to pandas DataFrame
        dataset_df.to_csv(csv_name, encoding='utf-8')
        print(f"Dataset saved to CSV at: {csv_name}")
        return csv_name
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def csv_to_json(csv_file):
    try:
        # Load CSV file into DataFrame
        data = pd.read_csv(csv_file)
        print("Loaded CSV data:", data)

        # Convert the DataFrame to a JSON format
        json_file = os.path.splitext(csv_file)[0] + ".json"
        
        # Convert to dictionary and write to JSON
        data_dict = data.to_dict(orient='records')  # Convert to list of dictionaries
        with open(json_file, 'w', encoding='utf-8') as json_f:
            json.dump(data_dict, json_f, ensure_ascii=False, indent=4)
        
        print(f"JSON saved to: {json_file}")
        return json_file
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def load_json_for_finetuning(json_file):
    try:
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict(data)
        
        print("Loaded dataset for fine-tuning:", dataset)
        return dataset
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def csv_to_huggingface_dataset(csv_file):
    try:
        data = pd.read_csv(csv_file)
        data.columns = data.columns.str.strip()

        print("Loaded data:")
        print(data.head())
        print(data.dtypes)
        
        required_columns = ['Question', 'Input', 'Answer', 'Reference']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"The following required columns are missing: {', '.join([col for col in required_columns if col not in data.columns])}")

        features = Features({
            'Question': Value(dtype='string'),
            'Input': Value(dtype='string'),
            'Answer': Value(dtype='string'),
            'Reference': Value(dtype='string') 
        }) 
        
        dataset = Dataset.from_pandas(data, features=features)        
        print(f"Converted to Hugging Face Dataset:", dataset)
        return dataset

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def preprocess_function(examples, tokenizer):
    text = examples['Question'] + ", " + examples['Input'] + ", " + examples['Answer'] + ", " + examples['Reference']
    return tokenizer(text, padding="max_length", truncation=True)

##
##  fine-tune LoRA
##

def train_finetune_lora(model_path, 
                  dataset_path, 
                  save_path, 
                  lora_r, 
                  lora_alpha, 
                  lora_dropout,
                  learning_rate, 
                  batch_size, 
                  epochs, 
                  device, 
                  ):
    try:
        if device == 'cuda':
            c_device = "cuda"
            deviceused = False
        else:
            c_device = "cpu"
            deviceused = True

        print(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, load_in_8bit=False)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print(f"second {c_device} | device used {deviceused}")
        
        model.to(torch.device(c_device))
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)

        #dataset_path = dataset_path.replace("\\", "/")
        #dataset = load_dataset('csv', data_files=dataset_path)['train']
        
        #dataset = csv_to_huggingface_dataset(dataset)

        json_file = csv_to_json(dataset_path)

        dataset = load_json_for_finetuning(json_file)

        dataset_split = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset_split['train']
        #valid_dataset = dataset_split['test']

        print(f"Train Dataset Type: {type(train_dataset)}")
        #print(f"Validation Dataset Type: {type(valid_dataset)}")

        train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        #valid_dataset = valid_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        training_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_dir='./logs',
            save_steps=500,
            load_best_model_at_end=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",      
            save_total_limit=3,
            use_cpu=deviceused,
            fp16=False,
            dataloader_num_workers=0
        )

        #preprocess_function(tokenizer, examples)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            #eval_dataset=valid_dataset,
            tokenizer=tokenizer
        )

        print("Fine-tuning started...")
        trainer.train()

        print(f"Saving fine-tuned model to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        #return save_path

    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return None

##
##  few-show RAG
##

from datasets import Dataset
import random

def few_shot_rag(dataset_path, few_shot_examples=5):
    """
    Prepare a few-shot dataset for Retrieval-Augmented Generation (RAG).

    Args:
        dataset_path (str): Path to the dataset in JSON format.
        few_shot_examples (int): Number of examples to sample for few-shot learning.

    Returns:
        Dataset: A few-shot subset of the dataset.
    """
    # Load the full dataset
    dataset = load_dataset("json", data_files=dataset_path)["train"]

    # Shuffle and select a few-shot subset
    few_shot_dataset = dataset.shuffle(seed=42).select(range(few_shot_examples))

    print(f"Few-shot dataset with {few_shot_examples} examples prepared.")
    return few_shot_dataset

##
##  fine-tune BERT
##

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the pre-trained WangchanBERTa model and tokenizer
model_name = "model/wangchanberta"

def finetune_bert(model_name, 
                  save_path, 
                  csv_path,
                  bert_lr=2e-5, 
                  bert_tr_batchsize=8, 
                  bert_ev_batchsize=8, 
                  train_epoch=10, 
                  w_decay=0.01
                  ):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, load_in_8bit=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=True)  # binary classification

    # Prepare your dataset
    # Replace with your dataset loading and preprocessing
    dataset = load_dataset(csv_path)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["question"], examples["input_answer"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set up the Trainer
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        learning_rate=bert_lr,
        per_device_train_batch_size=bert_tr_batchsize,
        per_device_eval_batch_size=bert_ev_batchsize,
        num_train_epochs=train_epoch,
        weight_decay=w_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()


import io, base64

import matplotlib.pyplot as plt
import seaborn as sns

def plot_lora(lora_metrics, metric_name="Accuracy", plot_folder="plot_folder", model_name="model_name"):
    """
    Plots the performance metrics (e.g., Accuracy, Loss) for LoRA.    
    Args:
    lora_metrics (dict): Metrics for LoRA with keys 'epochs' and 'values'.
    metric_name (str): Name of the metric to plot (e.g., "Accuracy", "Loss").
    """
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    # Create the full path for the plot file
    plot_path = os.path.join(plot_folder, f"{model_name}.png")

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(lora_metrics['epochs'], lora_metrics['values'], label="LoRA", color='blue', linestyle='-', marker='o')
    plt.title(f"LoRA {metric_name} Over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(f"{metric_name}", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)  # Move the cursor to the beginning of the BytesIO object
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()  # Close the plot to avoid it from displaying in an interactive session

    return img_base64

def plot_bert(bert_metrics, metric_name="Accuracy"):
    """
    Plots the performance metrics (e.g., Accuracy, Loss) for BERT.    
    Args:
    bert_metrics (dict): Metrics for BERT with keys 'epochs' and 'values'.
    metric_name (str): Name of the metric to plot (e.g., "Accuracy", "Loss").
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(bert_metrics['epochs'], bert_metrics['values'], label="BERT", color='green', linestyle='-', marker='x')
    plt.title(f"BERT {metric_name} Over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(f"{metric_name}", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rag(rag_metrics, metric_name="Accuracy"):
    """
    Plots the performance metrics (e.g., Accuracy, Loss) for Few-shot RAG.
    Args:
    rag_metrics (dict): Metrics for Few-shot RAG with keys 'epochs' and 'values'.
    metric_name (str): Name of the metric to plot (e.g., "Accuracy", "Loss").
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(rag_metrics['epochs'], rag_metrics['values'], label="Few-shot RAG", color='red', linestyle='-', marker='s')
    plt.title(f"Few-shot RAG {metric_name} Over Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel(f"{metric_name}", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
# Example usage
lora_metrics = {'epochs': [1, 2, 3, 4, 5], 'values': [0.65, 0.70, 0.75, 0.80, 0.85]}  # Accuracy for LoRA
bert_metrics = {'epochs': [1, 2, 3, 4, 5], 'values': [0.60, 0.68, 0.72, 0.78, 0.82]}  # Accuracy for BERT
rag_metrics = {'epochs': [1, 2, 3, 4, 5], 'values': [0.55, 0.65, 0.70, 0.74, 0.80]}  # Accuracy for RAG

# Plot each model's performance separately
plot_lora(lora_metrics, metric_name="Accuracy")
plot_bert(bert_metrics, metric_name="Accuracy")
plot_rag(rag_metrics, metric_name="Accuracy")
"""