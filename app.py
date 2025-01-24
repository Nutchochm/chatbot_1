from flask import Flask, render_template, request, jsonify
from chat import gen_response, load_model, loadtyphoon2, typhoon2chat
from chkans_app import load_chkans_model, tokenize_input, get_embedding, compute_similarity, rating_scores
from app_training_technique import train_finetune_lora, excel_csv, finetune_bert, few_shot_rag
from pymongo import MongoClient
from datetime import datetime
from pytz import timezone
import re
import json
import os
import pandas as pd


app = Flask(__name__)

# MongoDB Connection
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["chatbot_db"]
    messages_collection = db["messages"]
    topic_collection = db["topics"]
    users_collection = db["users"]
    print("Connected to MongoDB!")
except Exception as e:
    print(f"Could not connect to MongoDB: {e}")

# Timezone
bkk_tz = timezone("Asia/Bangkok")  

# Define default configuration
default_config = {
    "guest": {
        "role": 0,
        "folder": "",
        "radioID": 0
        },
    "user": {
        "role": 1,
        "folder": "",
        "radioID": 1
        },
    "chkans": {
        "role": 2,
        "folder": "",
        "activate": 0
    }    
}


@app.route("/chat")
def chat():
    topics = topic_collection.distinct("topic_name")
    return render_template("chat.html", topic=topics)


@app.route('/activate_model', methods=['POST'])
def initialize():
    expected_token = load_token()
    data = request.get_json()
    configmodel = load_adminconfig(CONFIG_FILE)

    # Extract token from received data
    received_token = data.get("token")
    auth_value = data["role"]
    user_query = data.get("answer")        
    user_id = data.get("user_id")
    user_name = data.get("user_name")

    if auth_value == 0:
        adminconfig_model_name = configmodel.get("guest", {}).get("folder")
    elif auth_value == 1:
        adminconfig_model_name = configmodel.get("user", {}).get("folder")
    else:
        return jsonify({"error": "Invalid role."}), 400


    if received_token != expected_token: 
        return jsonify({"error": "Invalid token."}), 401
    else:
        typhoon2_model, typhoon2_tokenizer = loadtyphoon2(adminconfig_model_name)
                
        data = {
            "answer": user_query,
            "role": auth_value,
            "token": received_token, 
            "user_id": user_id,
            "user_name": user_name
        }

        topic_name = re.split(r'[.!?]', user_query)[0].strip()
        if not topic_name:
            topic_name = user_query.split()[0]
            data["topic_name"] = topic_name

        response = typhoon2chat(user_query, typhoon2_model, typhoon2_tokenizer)
        message_id, topic_id = get_next_id(user_id, user_name, topic_name)

        insert_to_users(user_id, user_name, topic_id)
        insert_to_message(message_id, user_id, topic_id, topic_name, user_query, response)                
        
        return jsonify({
            "response": response,
        }), 200

def load_token():
    with open("appconfig.json", "r") as file:
        data = json.load(file)
        return data["token"]

def load_adminconfig(adminfile):
    try:
        # Check if file exists and can be read
        if not os.path.exists(adminfile):
            print(f"File not found: {adminfile}")
            with open(adminfile, 'w') as file:
                json.dump(default_config, file, indent=4)
            return default_config        
        with open(adminfile, 'r') as file:
            config = json.load(file)            
            
            if 'guest' not in config:
                config['guest'] = {"role": "guest", "folder": "", "radioID": 0}
            if 'user' not in config:
                config['user'] = {"role": "user", "folder": "", "radioID": 0}
            if 'chkans' not in config:
                config['chkans'] = {"role": "admin", "folder": "", "activate": 0}
            
            if 'guest' in config['guest']:
                config['guest']['radioID'] = int(config['guest'].get('radioID', 0))
            if 'user' in config['user']:
                config['user']['radioID'] = int(config['user'].get('radioID', 0))
            if 'chkans' in config['chkans']:
                config['chkans']['activate'] = int(config['chkans'].get('activate', 0))

            with open(adminfile, 'w') as file:
                json.dump(config, file, indent=4)
            return config        
    except (FileNotFoundError, json.JSONDecodeError):
        return default_config
    
def insert_to_users(user_id, user_name, topic_id):
    message = {
        "user_id": user_id,
        "user_name": user_name,
        "topic_id": topic_id,
        "created_at": datetime.now(bkk_tz)
    }
    users_collection.insert_one(message)

def insert_to_message(message_id, user_id, topic_id, topic_name, user_message, ai_message):
    message = {
        "message_id": message_id,
        "user_id": user_id,
        "topic_id": topic_id,
        "topic_name": topic_name,
        "user_message_id": f"user_{message_id}",
        "user_message": user_message,
        "ai_message_id": f"ai_{message_id}",
        "ai_message": ai_message,
        "timestamp": datetime.now(bkk_tz)
    }
    messages_collection.insert_one(message)
    
def insert_to_topic(user_id, user_name, topic_name):
    # Get the current month
    current_month = datetime.now(bkk_tz).strftime("%Y-%m")
    
    # Check the latest topic for the user
    last_topic = topic_collection.find_one(
        {"user_id": user_id, "user_name": user_name, "topic_name": topic_name},
        sort=[("created_at", -1)]
    )
    
    # Reset topic_id if the month has changed or no previous topic exists
    if not last_topic or last_topic["created_at"].strftime("%Y-%m") != current_month:
        new_topic_id = "0001"
    else:
        new_topic_id = str(int(last_topic["topic_id"]) + 1).zfill(4)
    
    # Insert the new topic
    topic_collection.insert_one({
        "user_id": user_id,
        "user_name": user_name,
        "topic_id": new_topic_id,
        "topic_name": topic_name,
        "message_seq": 1,
        "created_at": datetime.now(bkk_tz)
    })
    return new_topic_id

def get_next_id(user_id, user_name, topic_name):
    # Find an existing topic for the user and topic name
    existing_topic = topic_collection.find_one(
        {"user_id": user_id, "user_name": user_name, "topic_name": topic_name},
        sort=[("created_at", -1)]
    )
    
    # If no topic exists or the month has changed, insert a new topic
    if not existing_topic or existing_topic["created_at"].strftime("%Y-%m") != datetime.now(bkk_tz).strftime("%Y-%m"):
        topic_id = insert_to_topic(user_id, user_name, topic_name)
    else:
        topic_id = existing_topic["topic_id"]
    
    # Increment the message sequence for the topic
    topic = topic_collection.find_one_and_update(
        {"topic_name": topic_name, "user_name": user_name, "topic_id": topic_id},
        {"$inc": {"message_seq": 1}},
        return_document=True
    )
    if not topic:
        raise ValueError(f"Failed to update sequence for topic: {topic_name}")
    
    return str(topic["message_seq"]).zfill(4), topic_id

## ไม่ get ค่า ใน mongodb
@app.route('/chatbot_db/topics', methods=['GET'])
def get_topics():
    user_id = request.args.get('user_id')
    message_id = request.args.get('message_id')
    topic_id = request.args.get('topic_id')
    topic_name = request.args.get('topic_name')

    # Find topics related to the user (assuming topics have user_id or user_name)
    topics = list(topic_collection.find({"user_id": user_id, 
                                         "message_id": message_id,
                                         "topic_id": topic_id,
                                         "topic_name": topic_name
                                         }))
    topic_names = [topic['topic_name'] for topic in topics]
    
    return jsonify({"topics": topic_names}), 200

# Route to fetch messages for a topic
@app.route('/chatbot_db/messages/<topic_name>', methods=['GET'])
def get_messages(topic_name):
    user_id = request.args.get('user_id')

    # Fetch messages for the given topic and user_id
    messages = list(messages_collection.find({"topic_name": topic_name, "user_id": user_id}))
    
    # Sort by 'user_id' and 'ai_message' (adjust sort fields as per your schema)
    messages.sort(key=lambda msg: (msg.get("user_id"), msg.get("ai_message")))
    
    # Convert _id to string for JSON serialization
    for message in messages:
        message["_id"] = str(message["_id"])

    return jsonify({"topic_name": topic_name, "messages": messages}), 200

#########################################################################################################
################################              Training Page              ################################
#########################################################################################################

@app.route("/training")
def training():
    return render_template("training.html")

UPLOAD_DATASET = 'upload_dataset'
UPLOAD_MODEL = 'upload_model'

# 1. Upload dataset
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    try:
        if not os.path.exists(UPLOAD_DATASET):
            os.makedirs(UPLOAD_DATASET, exist_ok=True)
        
        dataset = request.files.get("dataset")
        if not dataset:
            return jsonify({"success": False, "message": "No dataset file provided."})

        # Save file locally
        dataset_path = os.path.join(UPLOAD_DATASET, dataset.filename)
        dataset.save(dataset_path)
        print(f"Dataset uploaded to: {dataset_path}")

        return jsonify({"success": True, "message": "Dataset uploaded successfully.", "dataset_path": dataset_path, "filename": dataset.filename})

    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})
        

# 2. Upload model or load models from a directory
@app.route("/upload_model", methods=["POST"])
def upload_model():
    try:
        # Ensure upload directory exists
        if not os.path.exists(UPLOAD_MODEL):
            os.makedirs(UPLOAD_MODEL)
        
        # Get the model folder name from the form
        model_folder_name = request.form.get("model_folder_name")
        if not model_folder_name:
            return jsonify({"success": False, "message": "Model folder name not provided."})
        
        # Create the model-specific folder if it doesn't exist
        model_folder_path = os.path.join(UPLOAD_MODEL, model_folder_name)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        # Retrieve the expected file names from the frontend
        file_names = request.form.get("file_names")
        if not file_names:
            return jsonify({"success": False, "message": "No filenames provided."})

        expected_files = json.loads(file_names)

        # Retrieve the uploaded files from the request
        files = request.files.getlist("model_files[]")
        if not files:
            return jsonify({"success": False, "message": "No files provided."})

        # List the uploaded filenames
        uploaded_files = [os.path.basename(file.filename) for file in files]

        # Validate if the uploaded files match the expected filenames
        if set(uploaded_files) != set(expected_files):
            return jsonify({"success": False, "message": "Filename mismatch."})

        # Save files into the model folder
        saved_files = []
        for file in files:
            try:
                if file and file.filename:                                      
                    # Define the path where the file will be saved
                    file_path = os.path.join(model_folder_path, os.path.basename(file.filename))
                    file.save(file_path)
                    saved_files.append(file.filename)
                    print(f"Model File is uploaded to: {file_path}")
            except Exception as e:
                print(f"Error saving file {file.filename}: {str(e)}")

        return jsonify({"success": True, "message": "Files uploaded successfully.", "files": saved_files, "model_folder_path": model_folder_path})

    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})


@app.route('/finetune_lora', methods=['POST', 'GET'])
def finetune_lora():
    try:
        dataset_path = request.form.get('dataset_path')
        newmodel_path = request.form.get('output_path')
        newmodel_name = request.form.get('model_name') 
        
        print(f"Received dataset_path: {dataset_path}")
        print(f"Received output_path: {newmodel_path}")
        print(f"Received model_name: {newmodel_name}")


        if not newmodel_name or not dataset_path or not newmodel_path:
            return jsonify({"success": False, "message": "Missing required fields!"})

        if not os.path.exists(dataset_path):
            return jsonify({"success": False, "message": f"Dataset path does not exist: {dataset_path}"})

        if not os.path.exists(newmodel_path):
            os.makedirs(newmodel_path, exist_ok=True)

        csvform = excel_csv(dataset_path)

        save_model_dir = os.path.join(newmodel_path, newmodel_name)
        os.makedirs(save_model_dir, exist_ok=True)

        # Log the save path
        print(f"Model will be saved at: {save_model_dir}")

        lora_r = request.form.get('lora_r')
        lora_alpha = request.form.get('lora_alpha')
        lora_dropout = request.form.get('lora_dropout')
        learning_rate = request.form.get('learning_rate')
        batch_size = request.form.get('batch_size')
        epochs = request.form.get('epochs')
        receive_device = request.form.get('device')
        print(f"first {receive_device}")

        modelupload_folder_name = request.form.get("model_folder_name")
        print("Form keys received:", request.form.keys())
        print(f"Model upload folder name: {modelupload_folder_name}")
        
        # Create the model-specific folder if it doesn't exist
        model_folder_path = os.path.join(UPLOAD_MODEL, modelupload_folder_name)
        if not os.path.exists(model_folder_path):
            return jsonify({"success": False, "message": "Please upload base model"})
        
        
        # Call the finetune_lora function with the parameters
        train_finetune_lora(
            model_path=model_folder_path,
            dataset_path=csvform,
            save_path=save_model_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            device=receive_device
        )       
        
        return jsonify({"success": True, "message": f"Training completed! Model saved to: {save_model_dir}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})


app.route("/finetune_bert")
def finetune_bert():
    try:
        dataset_path = request.form.get('dataset_path')
        newmodel_path = request.form.get('output_path')
        newmodel_name = request.form.get('model_name') 
        
        print(f"Received dataset_path: {dataset_path}")
        print(f"Received output_path: {newmodel_path}")
        print(f"Received model_name: {newmodel_name}")


        if not newmodel_name or not dataset_path or not newmodel_path:
            return jsonify({"success": False, "message": "Missing required fields!"})

        if not os.path.exists(dataset_path):
            return jsonify({"success": False, "message": f"Dataset path does not exist: {dataset_path}"})

        if not os.path.exists(newmodel_path):
            os.makedirs(newmodel_path, exist_ok=True)

        csvform = excel_csv(dataset_path)

        save_model_dir = os.path.join(newmodel_path, newmodel_name)
        os.makedirs(save_model_dir, exist_ok=True)

        # Log the save path
        print(f"Model will be saved at: {save_model_dir}")

        bert_lr = int(request.form.get('bert_lr', 2e-5))
        bert_tr_batchsize = int(request.form.get('bert_tr_batchsize', 8))
        bert_ev_batchsize = float(request.form.get('bert_ev_batchsize',8))
        train_epoch = int(request.form.get('train_epoch', 10))
        w_decay = float(request.form.get('w_decay', 0.01))
        receive_device = request.form.get('device')
        print(f"first {receive_device}")

        modelupload_folder_name = request.form.get("model_folder_name")
        print(f"model upload name : {modelupload_folder_name}")
        
        # Create the model-specific folder if it doesn't exist
        model_folder_path = os.path.join(UPLOAD_MODEL, modelupload_folder_name)
        if not os.path.exists(model_folder_path):
            return jsonify({"success": False, "message": "Please upload base model"})

        save_bert = finetune_bert(model_name=model_folder_path, 
                  save_path=save_model_dir, 
                  csv_path=csvform,
                  bert_lr=bert_lr, 
                  bert_tr_batchsize=bert_tr_batchsize, 
                  bert_ev_batchsize=bert_ev_batchsize, 
                  train_epoch=train_epoch, 
                  w_decay=w_decay,
                  device=receive_device
                  )
    

        return jsonify({"success": True, "message": f"Training completed! Model saved to: {save_bert}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})
    

############################################################################################
##########################              admin config              ##########################
############################################################################################


CONFIG_FILE = "configmodel.json"
MODEL_TRAIN_FOLDER = os.path.join(os.getcwd(), 'model_trained')
print(f"MODEL_FOLDER path: {MODEL_TRAIN_FOLDER}")

@app.route("/adminconfig")
def admin_config():
    try:        
        max_num_radio = 2
        selected_radio_buttons = {}
        # Check if MODEL_FOLDER exists and is a directory
        if os.path.exists(MODEL_TRAIN_FOLDER) and os.path.isdir(MODEL_TRAIN_FOLDER):
            folder_names = [f for f in os.listdir(MODEL_TRAIN_FOLDER) if os.path.isdir(os.path.join(MODEL_TRAIN_FOLDER, f))]
        else:
            folder_names = ["Directory not found."]
    except FileNotFoundError:
        folder_names = ["Directory not found."]
    except Exception as e:
        folder_names = [f"Error: {e}"]

    admin_config_data  = load_adminconfig(CONFIG_FILE)
    print("admin config", admin_config_data)

    # Mapping folder names to available radio button values (1 to max_num_radio)
    folder_radio_map = {folder: list(range(0, max_num_radio)) for folder in folder_names}

    return render_template(
        "adminconfig.html",
        folders=folder_names,
        max_num_radio=max_num_radio,
        folder_radio_map=folder_radio_map,
        selected_radio_buttons=selected_radio_buttons,
        admin_config=admin_config_data
    )

@app.route('/submit_adminconfig', methods=['POST'])
def submit_adminconfig():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No data received in request.")
        print("Received data:", data) 

        admin_config = load_adminconfig(CONFIG_FILE)
        print("Loaded admin config before update:", admin_config)

        admin_config.update(data)

        with open(CONFIG_FILE, 'w') as config_file:
            json.dump(admin_config, config_file, indent=4)

        return jsonify({"message": "Config saved successfully", "redirect": "/adminconfig"}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process the request"}), 400


########################################################################
##############                check_answer                ##############
########################################################################

@app.route('/checker', methods=['POST'])
def check_answer():
    data = request.get_json()

    input_question = data.get("input_question")
    correct_answer = data.get("correct_answer")
    input_answer = data.get("input_answer")
    received_token = data.get("token")

    if not input_question or not correct_answer or not input_answer:
        return jsonify({"error": "Missing required fields."}), 400
    
    expected_token = load_token()
    
    if received_token != expected_token: 
        return jsonify({"error": "Invalid token."}), 401
    
    ans_models = get_ans_models(CONFIG_FILE)
    print("check answer model named : ", ans_models)  
        
    if not ans_models:
        return jsonify({"error": "No valid model folder found in configuration."}), 500
    
    model_name = os.path.join(MODEL_TRAIN_FOLDER, ans_models)
    if not os.path.exists(model_name):
        print(f"Model not found: {model_name}")

    ca_model, ca_tokenizer = load_chkans_model(model_name)

    input_encodings = tokenize_input(input_question, input_answer, ca_tokenizer)
    correct_encodings = tokenize_input(input_question, correct_answer, ca_tokenizer)
    
    input_embedding = get_embedding(input_encodings, ca_model)
    correct_embedding = get_embedding(correct_encodings, ca_model)

    print(correct_answer)
    print(input_answer)
    
    similarity_score = compute_similarity(input_embedding, correct_embedding)
    print(f"Similarity Score: {similarity_score:.2f}")

    ratings = rating_scores(similarity_score)

    # Threshold for human-acceptable similarity
    if similarity_score >= 0.8:        
        print(f"Answer is correct or highly similar. given score: {ratings}")
    elif 0.5 <= similarity_score < 0.8:
        print("Answer is partially correct.", ratings)
    else:
        print("Answer is incorrect.", ratings)

    return jsonify({"Similarity Score": similarity_score, "Ratings": ratings})
    

def get_ans_models(config_file):
    if not os.path.exists(config_file):
        return {"error": f"Config file {config_file} does not exist."}

    try:
        with open(config_file, "r") as file:
            config = json.load(file)
            print(f"Loaded config: {config}")
            
            if "chkans" not in config or not isinstance(config["chkans"], dict):
                return {"error": "'chkans' is missing or not a list."}
            
            chkans = config["chkans"]
            ans_model = chkans.get("folder")
            
            if not ans_model:
                return {"error": "'folder' key is missing in 'chkans'."}
            
            return ans_model

    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
