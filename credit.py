import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

data =pd.read_csv("train.csv")

#show the data that is done
print(data.head(10))

#showing information about the columns
print(data.info())

#check if the dataset train.csv has any null values
print(data.isnull().sum())

#Looking at the Credit_Score column values:

print("\n")
print(data["Credit_Score"].value_counts())

#dataset exploration
#exploring to see if their occupation affects their credit scores

fig =px.box(data,
            x ="Occupation",
            color="Credit_Score",
            title ="Credit Scores Based on Occupation",
            color_discrete_map={'Poor':'red',
                                'Standard': 'yellow',
                                'Good':'green'})

fig.show()

#Let reviewing it the annual income of a person impacts credit scores

fig = px.box(data,
            x="Credit_Score",
            y="Annual_Income",
            color ="Credit_Score",
            title="Credit Scores Based on Annual Income",
            color_discrete_map={'Poor':'red',
                                'Standard':'yellow',
                                    'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#reviewing if bank accounts have an impacts on credit score

fig=px.box(data,
        x="Credit_Score",
        y="Num_Bank_Accounts",
        color ="Credit_Score",
        title="Credit Scores Based on Number of Bank Accounts",
        color_discrete_map={'Poor':'red',
                            'Standard':'yellow',
                            'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()

#reviewing the impact on credit scores based on the number of credit cards you have

fig = px.box(data, 
             x="Credit_Score", 
             y="Num_Credit_Card", 
             color="Credit_Score",
             title="Credit Scores Based on Number of Credit cards", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()


#reviewing the affects on credit score on the interest paid in loans
fig = px.box(data, 
             x="Credit_Score", 
             y="Interest_Rate", 
             color="Credit_Score",
             title="Credit Scores Based on the Average Interest rates", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#lets review the number of loans you can take depending on credit score

fig=px.box(data,
            x="Credit_Score",
            y="Num_of_Loan",
            color="Credit_Score",
            title="Credit Scores Based on Number of Loans Taken by a Person",
            color_discrete_map={'Poor':'red',
                                'Standard':'yellow',
                                'Good':'green'})

fig.update_traces(quartilemethod="exclusive") 
fig.show()

#reviewing if delaying payaments on credit affects credit score

fig =px.box(data,
            x="Credit_Score",
            y="Delay_from_due_date",
            color="Credit_Score",
            title="Credit Scores Based on Average Number of Days Delayed for Credit card Payments", 
            color_discrete_map={'Poor':'red',
                                'Standard':'yellow',
                                'Good':'green'})

fig.update_traces(quartilemethod="exclusive")
fig.show()

#reviweing if frequents delaying your credit payments will affects your credit score

fig = px.box(data, 
             x="Credit_Score", 
             y="Num_of_Delayed_Payment", 
             color="Credit_Score", 
             title="Credit Scores Based on Number of Delayed Payments",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#reviewing if having more debts will affect your credit score
fig = px.box(data, 
             x="Credit_Score", 
             y="Outstanding_Debt", 
             color="Credit_Score", 
             title="Credit Scores Based on Outstanding Debt",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#reveing if having a high credit utilization ratio will affect credit scores or not

fig = px.box(data, 
             x="Credit_Score", 
             y="Credit_Utilization_Ratio", 
             color="Credit_Score",
             title="Credit Scores Based on Credit Utilization Ratio", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#reviweing how how the credit history age of a person affects credit scores
fig = px.box(data, 
             x="Credit_Score", 
             y="Credit_History_Age", 
             color="Credit_Score", 
             title="Credit Scores Based on Credit History Age",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#reviewing how many EMIs you can have in a month for a good credit score
fig = px.box(data, 
             x="Credit_Score", 
             y="Total_EMI_per_month", 
             color="Credit_Score", 
             title="Credit Scores Based on Total Number of EMIs per Month",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#reviewing how  monthly investments affect your credit scores or not
fig = px.box(data, 
             x="Credit_Score", 
             y="Amount_invested_monthly", 
             color="Credit_Score", 
             title="Credit Scores Based on Amount Invested Monthly",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#reviewing  of low amount at the end of the month affects credit scores or not
fig = px.box(data, 
             x="Credit_Score", 
             y="Monthly_Balance", 
             color="Credit_Score", 
             title="Credit Scores Based on Monthly Balance Left",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

#Credit Score Classification Model
data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, 
                               "Good": 2, 
                               "Bad": 0})
#Now split the data into features and labels by selecting the features we found important for the model
from sklearn.model_selection import train_test_split
x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(data[["Credit_Score"]])

#split the data into training and test sets 
# and proceed further by training a credit score classification model

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.33, 
                                                    random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

#make predictions from our model by giving inputs to our model according to the features we used to train the model
print("\n")
print("=========================")
print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))
