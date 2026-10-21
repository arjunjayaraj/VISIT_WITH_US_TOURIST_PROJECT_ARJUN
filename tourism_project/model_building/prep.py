# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

# 1. Gender: fix the typo 'Fe Male' -> 'Female'
gender_key = (df["Gender"].str.strip().str.lower().str.replace(r"\s+", "", regex=True))
df["Gender"] = gender_key.map({"male": "Male", "female": "Female"})

# 1. MaritalStatus: data dictionary only defines Single / Married / Divorced, so mergin unmarried to Single
df["MaritalStatus"] = (df["MaritalStatus"].str.strip().str.title()
                       .replace({"Unmarried": "Single"}))

# Define the target variable for the classification task
target = "ProdTaken"

# List of numerical features in the dataset
numeric_features = [
    "ProdTaken",                 # Customer's credit score
    "Age",                       # Customer's age
    "CityTier",                  # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    "DurationOfPitch",           # Duration of the sales pitch delivered to the customer.
    "NumberOfPersonVisiting",    # Total number of people accompanying the customer on the trip.
    "NumberOfFollowups",         # Total number of follow-ups by the salesperson after the sales pitch.
    "PreferredPropertyStar",     # Preferred hotel rating by the customer.
    "NumberOfTrips",             # Average number of trips the customer takes annually.
    "Passport",                  # Whether the customer holds a valid passport (0: No, 1: Yes).
    "PitchSatisfactionScore",    # Score indicating the customer's satisfaction with the sales pitch.
    "OwnCar"                     # Whether the customer owns a car (0: No, 1: Yes).
    "NumberOfChildrenVisiting"   # Number of children below age 5 accompanying the customer.
    "MonthlyIncome"              # Gross monthly income of the customer.
]

# List of categorical features in the dataset
categorical_features = [
    "TypeofContact",         # The method by which the customer was contacted (Company Invited or Self Inquiry).
    "Occupation",            # Customer's occupation (e.g., Salaried, Freelancer).
    "Gender",                # Gender of the customer (Male, Female).
    "ProductPitched"         # The type of product pitched to the customer.
    "MaritalStatus"          # Marital status of the customer (Single, Married, Divorced).
    "Designation"            # Customer's designation in their current organization.
]

# Define predictor matrix (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y = df[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y,        # keeps the (imbalanced) churn ratio consistent across splits
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
