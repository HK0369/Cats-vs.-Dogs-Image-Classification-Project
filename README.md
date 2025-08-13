Cats vs. Dogs: End-to-End Image Classification Project
This repository contains the complete code for a deep learning project that builds, trains, and deploys a Convolutional Neural Network (CNN) to classify images of cats and dogs. The project culminates in a user-friendly web application built with Streamlit.

<!-- Replace with a screenshot of your running app -->

🚀 Key Features
End-to-End Workflow: Covers the entire machine learning pipeline from data preparation to a functional web application.

Deep Learning Model: Implements a CNN using TensorFlow and Keras to learn features from images.

Data Augmentation: Uses techniques like random rotations, zooms, and flips to create a more robust model and prevent overfitting.

Smart Training: Employs ModelCheckpoint and EarlyStopping to save only the best version of the model and avoid unnecessary training time.

Interactive Web UI: A frontend built with Streamlit allows users to upload their own images and get real-time predictions.

🛠️ Tech Stack
Backend & Model: Python, TensorFlow, Keras

Frontend: Streamlit

Data Handling: NumPy, Pillow

Presentation Generation: python-pptx

📂 Repository Structure
cats_vs_dogs_project/
│
├── data/
│   ├── raw/          # Original, untouched dataset
│   └── processed/    # Organized train/val/test splits
│
├── models/
│   └── cats_vs_dogs_model.h5  # The trained model (tracked by Git LFS)
│
├── .gitattributes      # Git LFS tracking configuration
├── app.py              # The main Streamlit application file
├── ppt.py              # Script to generate the project presentation
└── requirements.txt    # Project dependencies

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

# Install the required packages
pip install -r requirements.txt

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

⚠️ A Note on Git LFS
This repository uses Git Large File Storage (LFS) to manage the model file (cats_vs_dogs_model.h5), which is over 100 MB. To clone this repository correctly, you must have Git LFS installed on your system.
