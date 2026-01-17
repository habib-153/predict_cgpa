# Bangladesh Student Performance Predictor

A machine learning application that predicts HSC (Higher Secondary Certificate) exam results for Bangladeshi students based on demographic and academic features.

## Project Overview

This project uses a Random Forest Regression model to predict student GPA scores on the HSC examination. The model is trained on a dataset of 2020 Bangladeshi students and deployed using a Gradio web interface for easy interaction.

## Dataset Information

### Dataset: Bangladesh Student Performance

- **Total Records:** 2,020 students
- **Target Variable:** `hsc_result` (GPA score ranging from 0 to 5)
- **Features:** 14 input features

### Features Description

**Demographic Features:**

- `gender` - Student's gender (M/F)
- `age` - Student's age (typically 17-19)
- `address` - Residential area (Urban/Rural)
- `famsize` - Family size (LE3: ≤3 members, GT3: >3 members)
- `Pstatus` - Parent's cohabitation status (Together/Apart)

**Parental Information:**

- `M_Edu` - Mother's education level (0-4 scale)
- `F_Edu` - Father's education level (0-4 scale)
- `M_Job` - Mother's occupation (Teacher, Health, Services, At_home, Other)
- `F_Job` - Father's occupation (Teacher, Health, Services, Farmer, Business)
- `relationship` - Primary guardian (mother/father/other)

**Student Factors:**

- `smoker` - Smoking status (Yes/No)
- `tuition_fee` - Annual tuition fee paid
- `time_friends` - Time spent with friends (1-5 scale)
- `ssc_result` - SSC (Secondary School Certificate) exam GPA

## Project Structure

```
Project_1/
│
├── app.py                                  # Gradio web application
├── rf_train.py                             # Model training script
├── bangladesh_student_performance.csv      # Training dataset
├── student_performance_model.pkl           # Trained model (generated)
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## Model Details

**Algorithm:** Random Forest Regressor

**Hyperparameters:**

- n_estimators: 100
- max_depth: 10
- min_samples_split: 2
- random_state: 42

**Preprocessing Pipeline:**

- **Numerical Features:** Median imputation + Standard scaling
- **Categorical Features:** Most frequent imputation + One-hot encoding

**Train-Test Split:** 75% training, 25% testing

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Project_1
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to train the model and generate the pickle file:

```bash
python rf_train.py
```

This will:

- Load and preprocess the dataset
- Train the Random Forest model
- Display model performance metrics (RMSE, R² Score, MAE)
- Save the trained model as `student_performance_model.pkl`

### Running the Web Application

Launch the Gradio interface:

```bash
python app.py
```

The application will start on `http://localhost:7860` (or another available port).

### Making Predictions

1. Open the web interface in your browser
2. Fill in the student information:
   - Select gender, address, family size, and parental status
   - Enter age
   - Set mother's and father's education levels (0-4)
   - Choose occupations for both parents
   - Select primary guardian
   - Indicate smoking status
   - Enter tuition fee amount
   - Set time spent with friends (1-5)
   - Enter SSC result (GPA)
3. Click Submit to get the predicted HSC GPA

## Model Performance

The model's performance is evaluated using:

- **RMSE** (Root Mean Squared Error) - Measures prediction error magnitude
- **R² Score** - Coefficient of determination (model fit quality)
- **MAE** (Mean Absolute Error) - Average absolute prediction error

Performance metrics are displayed during model training.

## Technologies Used

- **Machine Learning:** scikit-learn
- **Web Interface:** Gradio
- **Data Processing:** pandas, numpy
- **Model Persistence:** pickle

## Key Dependencies

- gradio==6.3.0
- scikit-learn==1.8.0
- pandas==2.3.3
- numpy==2.4.1
- joblib==1.5.3

## Future Improvements

- Add cross-validation for more robust model evaluation
- Implement feature importance visualization
- Add model comparison (test other algorithms)
- Include data exploration and visualization dashboard
- Add input validation and error handling
- Deploy to cloud platform (Hugging Face Spaces, etc.)

## License

This project is available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue in the repository.
