import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

Border = "-"*60
def LoadData():
    print(Border)
    print("Load the Dataset using pandas")
    print(Border)

    df = pd.read_csv("Loan_Default.csv")

    return df

def DataAnalysis(df):
    print(Border)
    print("Analyze the the Dataset using pandas")
    print(Border)

    print("Firts five records of Dataset : ")
    print(df.head())
    print(Border)

    print("Shape of Dataset : ")
    print(df.shape)
    print(Border)

    print("Columns of Dataset : ")
    print(df.columns)
    print(Border)

    print("Missing Values of Dataset : ")
    print(df.isnull().sum())
    print(Border)

def CategoricalToNumerical(df):
    print(Border)
    print("Convert Categorical Data into Numerical")
    print(Border)

    df["PreviousDefault"] = df["PreviousDefault"].map({"Yes" :1, "No" : 0 })

    df = pd.get_dummies(
        df,
        columns=["HomeOwnership"],
        dtype=int
    )
    df = df.dropna()

    return df

def SeparateVariables(df):
    print(Border)
    print("Separate X and Y variable")
    print(Border)

    X = df.drop("Default",axis=1)

    Y = df['Default']

    return X,Y

def SplitVariables(X,Y):
    print(Border)
    print("Split the Dataset into training and testing ")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

    return X_train,X_test,Y_train,Y_test

def FeatureScaling(X_train,X_test):
    print(Border)
    print("Scale the Dataset")
    print(Border)

    scaler =StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train,X_test, scaler

def ModelCreation():
    print(Border)
    print("Model Creation ")
    print(Border)

    model = MLPClassifier(
        hidden_layer_sizes=(32,16),
        activation='relu',
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    return model

def TrainModel(model,X_train,Y_train):
    print(Border)
    print("Model Training ")
    print(Border)

    model = model.fit(X_train,Y_train)

    return model

def TestingModel(model,X_test,Y_test):
    print(Border)
    print("Model Testing ")
    print(Border)

    Y_pred = model.predict(X_test)

    TestingAccuracy = accuracy_score(Y_test,Y_pred)*100

    print("Testing Accuracy : ",TestingAccuracy)

    return  Y_pred

def TrainingAccuracy(model,X_train,Y_train):
    print(Border)
    print("Display training Accuracy and number of iteration")
    print(Border)

    Y_pred = model.predict(X_train)

    TrainAccuracy = accuracy_score(Y_train,Y_pred)*100

    print("Training Accuracy : ",TrainAccuracy)

def GenerateConfusionMatrix(Y_test,Y_pred):
    print(Border)
    print("Generate Confusion Matrix ")
    print(Border)

    CM = confusion_matrix(Y_test,Y_pred)
    print("Confusion Matrix : ")
    print(CM)

def LossCurve(model):
    print(Border)
    print("Plot the Loss Curve")
    print(Border)

    plt.plot(model.loss_curve_)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("MLP training Loss Curve")
    plt.show()

def ActivationExperiment(X_train,X_test,Y_train,Y_test):
    print(Border)
    print("Activation Function Experiment")
    print(Border)

    activations = ["identity", "logistic", "tanh", "relu"]

    for activation in activations:
        model = MLPClassifier(
            hidden_layer_sizes=(32,16),
            activation=activation,
            solver= 'adam',
            max_iter=1000,
            random_state=42
        )

        model.fit(X_train,Y_train)
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test,Y_pred)* 100

        print("Activation : ",activation)
        print("Accuracy : ",accuracy)
        print(Border)

def HiddenLayersExperiments(X_train,X_test,Y_train,Y_test):
    print(Border)
    print("Hidden Layer Experiment")
    print(Border)

    hidden_layers = [
        (10,),
        (20,10),
        (50,25),
        (100,50,25)
    ]
    for layers in hidden_layers:
            model = MLPClassifier(
                hidden_layer_sizes=layers,
                activation="relu",
                solver= 'adam',
                max_iter=1000,
                random_state=42
            )
    
            model.fit(X_train,Y_train)
            Y_pred = model.predict(X_test)
            accuracy = accuracy_score(Y_test,Y_pred)* 100
    
            print("Hidden Layer : ",layers)
            print("Accuracy : ",accuracy)
            print(Border)

def LearningRateExperiment(X_train,X_test,Y_train,Y_test):
    print(Border)
    print("Learning Rate Experiment")
    print(Border)

    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    for rate in learning_rates:
            model = MLPClassifier(
                hidden_layer_sizes=(32,16),
                activation="relu",
                solver= 'adam',
                learning_rate_init=rate,
                max_iter=1000,
                random_state=42
            )
    
            model.fit(X_train,Y_train)
            Y_pred = model.predict(X_test)
            accuracy = accuracy_score(Y_test,Y_pred)* 100
    
            print("Learning Rate : ",rate)
            print("Accuracy : ",accuracy)
            print(Border)

def TestNewDataset(model, scaler):

    print(Border)
    print("Test the Model on New Loan Applicants")
    print(Border)

    new_df = pd.read_csv("NewLoanApplicants.csv")

    new_df["PreviousDefault"] = new_df["PreviousDefault"].map({
        "Yes": 1,
        "No": 0
    })

    new_df = pd.get_dummies(
        new_df,
        columns=["HomeOwnership"],
        dtype=int
    )

    expected_columns = [
        'Age',
        'Income',
        'LoanAmount',
        'CreditScore',
        'EmploymentYears',
        'ExistingLoans',
        'MonthlyDebt',
        'LoanTerm',
        'PreviousDefault',
        'HomeOwnership_Mortgage',
        'HomeOwnership_Own',
        'HomeOwnership_Rent'
    ]

    for column in expected_columns:
        if column not in new_df.columns:
            new_df[column] = 0

    X_new = new_df[expected_columns]

    X_new = scaler.transform(X_new)

    Y_pred = model.predict(X_new)

    new_df["Prediction"] = Y_pred

    new_df["Prediction"] = new_df["Prediction"].map({
        0: "No Default",
        1: "Default"
    })

    print("\nPrediction on New Loan Applicants:")
    print(new_df)

    return Y_pred


def main():
    df = LoadData()

    DataAnalysis(df)

    df = CategoricalToNumerical(df)

    X, Y = SeparateVariables(df)

    X_train,X_test,Y_train,Y_test = SplitVariables(X,Y)

    X_train,X_test, scaler = FeatureScaling(X_train,X_test)

    model = ModelCreation()

    model = TrainModel(model,X_train,Y_train)

    Y_pred = TestingModel(model,X_test,Y_test)

    TrainingAccuracy(model,X_train,Y_train)

    GenerateConfusionMatrix(Y_test,Y_pred)

    LossCurve(model)

    ActivationExperiment(X_train,X_test,Y_train,Y_test)

    HiddenLayersExperiments(X_train,X_test,Y_train,Y_test)

    LearningRateExperiment(X_train,X_test,Y_train,Y_test)

    New_Y_pred = TestNewDataset(model,scaler)

if __name__ == "__main__":
    main()