// Toggle training settings visibility based on selected technique
/*
document.getElementById("technique").addEventListener("change", function () {
    const trainingSettings = document.getElementById("trainingSettings");
    const trainingSettings_2 = document.getElementById("trainingSettings_2");
    trainingSettings.style.display = this.value === "Fine-Tuning Lora" ? "block" : "none";
    trainingSettings_2.style.display = this.value === "Fine-Tuning Bert" ? "block" : "none";
});
*/

document.addEventListener("DOMContentLoaded", function () {
    const techniqueSelect = document.getElementById("technique");
    const trainingSettings = document.getElementById("trainingSettings");
    const trainingSettings_2 = document.getElementById("trainingSettings_2");

    // Initially hide training settings
    trainingSettings.style.display = "none";
    trainingSettings_2.style.display = "none";

    // Toggle training settings visibility based on selected technique
    techniqueSelect.addEventListener("change", function () {
        if (this.value === "Fine-Tuning Lora") {
            trainingSettings.style.display = "block";
            trainingSettings_2.style.display = "none";
        } else if (this.value === "Fine-Tuning Bert") {
            trainingSettings.style.display = "none";
            trainingSettings_2.style.display = "block";
        } else {
            // Hide both if no technique is selected
            trainingSettings.style.display = "none";
            trainingSettings_2.style.display = "none";
        }
    });

    // Handle the case where no technique is selected initially
    if (techniqueSelect.value === "None") {
        trainingSettings.style.display = "none";
        trainingSettings_2.style.display = "none";
    }
});


document.getElementById("uploadForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const uploadStatus = document.getElementById("uploadStatus");
    uploadStatus.innerHTML = "Uploading...";

    const formData = new FormData();
    const dataset = document.getElementById("dataset").files[0];

    if (!dataset) {
        uploadStatus.innerHTML = "Please select a file to upload.";
        return;
    }

    document.getElementById("submittrain").disable = true;

    formData.append("dataset", dataset);

    try {
        const response = await fetch("/upload_dataset", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            uploadStatus.innerHTML = `<span style="color: green;">${result.message}</span>`;
            document.getElementById("submittrain").disable = false;
            document.getElementById("submittrain").dataset.filename = result.filename;
        } else {
            uploadStatus.innerHTML = `<span style="color: red;">${result.message}</span>`;
        }
    } catch (error) {
        uploadStatus.innerHTML = `<span style="color: red;">An error occurred: ${error.message}</span>`;
    }
});

document.addEventListener("DOMContentLoaded", function () {
    const inputSelector = document.getElementById("inputSelector");
    const fileMode = document.getElementById("fileMode");
    const folderMode = document.getElementById("folderMode");

    // Toggle between file and folder modes
    fileMode.addEventListener("change", function () {
        inputSelector.removeAttribute("webkitdirectory");
        inputSelector.setAttribute("multiple", "true");
    });

    folderMode.addEventListener("change", function () {
        inputSelector.setAttribute("webkitdirectory", "");
        inputSelector.removeAttribute("multiple");
    });

    document.getElementById("loadmodelForm").addEventListener("submit", function (event) {
        event.preventDefault(); 
        const uploadmodelStatus = document.getElementById("uploadmodelStatus");
        uploadmodelStatus.innerHTML = "Uploading...";

        const inputSelector = document.getElementById("inputSelector");
        const formData = new FormData();

        const modelFolderName = document.getElementById("inputSelector").files[0].webkitRelativePath.split("/")[0];
        formData.append("model_folder_name", modelFolderName);

        const expectedFiles = Array.from(inputSelector.files).map(file => file.name);
        formData.append("file_names", JSON.stringify(expectedFiles));
        
        const fileNames = Array.from(inputSelector.files).map(file => file.name);
        // Append filenames as a JSON string
        formData.append("file_names", JSON.stringify(fileNames));

        // Append files to FormData
        Array.from(inputSelector.files).forEach(file => formData.append("model_files[]", file));
       
        // Send files to the backend
        fetch("/upload_model", {
            method: "POST",
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    uploadmodelStatus.innerHTML = `<p style="color: green;">Upload successful! Files are processed.</p>`;
                } else {
                    uploadmodelStatus.innerHTML = `<p style="color: red;">Error: ${data.message}</p>`;
                }
            })
            .catch(err => console.error("Error uploading files:", err));
    });
});

// Handle training form submission
document.getElementById("submittrain").addEventListener("click", function (event) {
    event.preventDefault();

    const selectedTechnique = document.getElementById('technique').value;
        
    const modelFolderName = document.getElementById("inputSelector").files[0].webkitRelativePath.split("/")[0];
    //console.log("Model Folder Name:", modelFolderName);  // Debug log
   
    const datasetFilename = this.dataset.filename;
    const modelName = document.getElementById("model_name").value.trim();
    const modelPath = document.getElementById("output_path").value.trim();
    const device = document.getElementById("device").value.trim();

    // Validate all required fields
    if (!validateTrainingSettings()) {
        alert("Please fill in all the fields.");
        return;
    }

    const formData = new FormData(document.getElementById("trainForm"));

    formData.append("model_folder_name", modelFolderName); 

    formData.append('dataset_path', `upload_dataset/${datasetFilename}`);
    formData.append("model_name", modelName);
    formData.append("output_path", modelPath);
    if (selectedTechnique === "Fine-Tuning Lora") {
        formData.append('lora_r', '16');
        formData.append('lora_alpha', '32');
        formData.append('lora_dropout', '0.1');
        formData.append('learning_rate', '1e-4');
        formData.append('batch_size', '4');
        formData.append('epochs', '3');
        formData.append('device', device);
    
    
        // Send training request
        fetch("/finetune_lora", {
            method: "POST",
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {       
                    const uploadedModelFolder = document.getElementById("uploadedModelFolder");
                    if (uploadedModelFolder) {
                        uploadedModelFolder.value = data.model_folder_name;
                    } else {
                        console.warn("Element with id 'uploadedModelFolder' not found.");
                    }
                }    
                document.getElementById("loramessages").textContent = "Training started successfully.";
                console.log("Response:", data);
            })
            .catch(error => {
                console.error("Error:", error);
                alert("An error occurred during training.");
            });
    } else if (selectedTechnique === "Fine-Tuning Bert") {
        formData.append('bert_lr', '2e-5');
        formData.append('bert_tr_bs', '8');
        formData.append('bert_ev_bs', '8');
        formData.append('bert_epochs', '10');
        formData.append('bert_decay', '0.01');

        // Send Bert-specific training request
        fetch("/finetune_bert", {
            method: "POST",
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const uploadedModelFolder = document.getElementById("uploadedModelFolder");
                    if (uploadedModelFolder) {
                        uploadedModelFolder.value = data.model_folder_name;
                    } else {
                        console.warn("Element with id 'uploadedModelFolder' not found.");
                    }
                }
                document.getElementById("bertmessages").textContent = "Bert training started successfully.";
                console.log("Response:", data);
            })
            .catch(error => {
                console.error("Error:", error);
                alert("An error occurred during Bert training.");
            });
        }
    
});

function validateTrainingSettings() {
    const requiredFields = [
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "learning_rate",
        "batch_size",
        "epochs",
        "output_path",
        "model_name",
    ];

    return requiredFields.every(id => document.getElementById(id).value.trim() !== "");
}
