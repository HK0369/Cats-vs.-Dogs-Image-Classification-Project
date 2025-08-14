🚀 Project Title & Tagline
================================

**Project Title:** Image Classification and Presentation Automation using Streamlit and TensorFlow
**Tagline:** Automate your image classification and presentation tasks with ease using this intuitive Python project

📖 Description
==============

This Python project is designed to automate image classification and presentation tasks using Streamlit and TensorFlow. The project comprises several files that work together to preprocess images, train a machine learning model, and present the results in a user-friendly interface. The project is divided into five main stages: data preparation, initial training, further training, model evaluation, and presentation.

The project uses the Keras API in TensorFlow to build and train a convolutional neural network (CNN) model. The model is trained on a dataset of images and is capable of classifying new images into predefined categories. The project also uses Streamlit to create a user-friendly interface for users to input images and view the classification results.

The project is designed to be modular and extensible, allowing users to easily add new features and modify existing ones. The project is also well-documented, making it easy for users to understand and customize the code.

✨ Features
==========

1. **Image Classification:** The project is capable of classifying images into predefined categories using a trained CNN model.
2. **Presentation Automation:** The project uses Streamlit to create a user-friendly interface for users to input images and view the classification results.
3. **Data Preparation:** The project includes scripts to preprocess images, such as resizing and normalizing the data.
4. **Initial Training:** The project includes scripts to train the CNN model on a dataset of images.
5. **Further Training:** The project includes scripts to further train the CNN model using additional data.
6. **Model Evaluation:** The project includes scripts to evaluate the performance of the CNN model using metrics such as accuracy and loss.
7. **Presentation Customization:** The project allows users to customize the presentation of the classification results, including the layout and design of the slides.
8. **Error Handling:** The project includes error handling mechanisms to handle unexpected errors and exceptions.

🧰 Tech Stack
=============

| **Tech Stack** | **Version** |
| --- | --- |
| Python | 3.9.7 |
| TensorFlow | 2.5.0 |
| Keras | 2.4.3 |
| Streamlit | 1.6.0 |
| Pillow | 8.4.0 |
| OpenCV | 4.5.5.64 |

📁 Project Structure
==================

* `app.py`: The main application file that uses Streamlit to create the user interface.
* `1_data_preparation.py`: A script to preprocess images and prepare the data for training.
* `2_initial_training.py`: A script to train the CNN model on a dataset of images.
* `3_further_training.py`: A script to further train the CNN model using additional data.
* `4_model_evaluation.py`: A script to evaluate the performance of the CNN model.
* `ppt.py`: A script to create a presentation using the classification results.
* `requirements.txt`: A file containing the dependencies required to run the project.

⚙️ Setup and Installation
Follow these steps to set up the project environment on your local machine.

1. Prerequisites
Python 3.8+

Git and Git LFS installed.

2. Clone the Repository
Clone the repository to your local machine. Git LFS will automatically handle the large model file.

# Install Git LFS (if you haven't already)
git lfs install

# Clone the repository
git clone https://github.com/HK0369/Cats-vs.-Dogs-Image-Classification-Project.git
cd Cats-vs.-Dogs-Image-Classification-Project

3. Install Dependencies
It is highly recommended to use a virtual environment.

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (macOS/Linux)
source venv/bin/activate

⚙️ How to Run
=============

1. **Setup:** Install the required dependencies by running `pip install -r requirements.txt`.
2. **Environment:** Create a new environment using `conda create --name project-env` and activate it using `conda activate project-env`.
3. **Build:** Run the project using `python app.py`.
4. **Deploy:** Deploy the project to a cloud platform or a local server.


🚀 Usage
1. Data Preparation & Training (Optional)
The repository includes a pre-trained model. However, if you wish to train the model from scratch, you must first download the Kaggle "Cats and Dogs" dataset and place the train and test folders inside the data/raw/ directory.

Then, run the training scripts in order:

python 1_data_preparation.py
python 2_initial_training.py
python 3_further_training.py

2. Run the Streamlit Application
To start the web application, run the following command:

streamlit run app.py

This will open a new tab in your web browser where you can upload images and see the model's predictions.

3. Generate the Presentation
To generate the project presentation slides, run:

python ppt.py

This will create a Cats_vs_Dogs_Project_Detailed.pptx file in the root directory.

🧪 Testing Instructions
=====================

1. **Test:** Run the project using `python app.py`.
2. **Verify:** Verify that the project is working correctly by inputting an image and viewing the classification results.


I hope this README.md file meets your requirements! Let me know if you need any further modifications. 😊

⚠️ A Note on Git LFS
This repository uses Git Large File Storage (LFS) to manage the model file (cats_vs_dogs_model.h5), which is over 100 MB. To clone this repository correctly, you must have Git LFS installed on your system.

