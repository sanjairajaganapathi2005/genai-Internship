import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('alumni_data.csv')

# Example preprocessing step: Create question-answer pairs from the dataset
qa_data = []

for index, row in data.iterrows():
    # Add academic score question-answer pairs
    qa_data.append({
        "question": f"What is the academic score of {row['Student Full Name']}?",
        "answer": f"{row['Academic Score %']}%"
    })
    
    # Add job offer question-answer pairs
    qa_data.append({
        "question": f"Did {row['Student Full Name']} get a job offer in campus placement?",
        "answer": row['Got Job Offer in Campus Placement']
    })
    
    # Add graduation year question-answer pairs
    qa_data.append({
        "question": f"What is the graduation year of {row['Student Full Name']}?",
        "answer": str(row['Graduation Year'])
    })
    
    # Add contact email question-answer pairs
    qa_data.append({
        "question": f"What is the contact email of {row['Student Full Name']}?",
        "answer": row['Contact Email']
    })
    
    # Add attendance percentage question-answer pairs
    qa_data.append({
        "question": f"What is the attendance percentage of {row['Student Full Name']}?",
        "answer": f"{row['Attendance %']}%"
    })

# Convert to a Hugging Face dataset
qa_df = pd.DataFrame(qa_data)
qa_dataset = Dataset.from_pandas(qa_df)

# Load pretrained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['question'], examples['answer'], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = qa_dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./deepseek_model")
tokenizer.save_pretrained("./deepseek_model")

# Function to query the trained model
def query_model(question):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(inputs['input_ids'], max_length=128)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example query
while(True):
    query = input("query : ")
    if(query=="exit"):
        break
    response = query_model(query)
    print(response)
