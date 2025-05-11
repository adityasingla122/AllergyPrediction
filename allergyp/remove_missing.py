import math
import pandas as pd
import numpy as np
from scipy.stats import linregress


# Define the names of the questions
names = ["MCQ010", "MCQ025", "MCQ035", "MCQ040", "MCQ050", "MCQ053", "MCQ080", "MCQ092", "MCD093", "MCQ140", "MCQ149", "MCQ150G", "MCQ150Q", "MCQ160A", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160G", "MCQ160K", "MCQ160L", "MCQ160M", "MCQ170K", "MCQ170M", "MCQ180A", "MCQ180B", "MCQ180C", "MCQ180D", "MCQ180E", "MCQ180F", "MCQ180G", "MCQ180K", "MCQ180L", "MCQ180M", "MCQ190", "MCQ220", "MCQ230A", "MCQ230B", "MCQ245A", "MCQ245B", "MCQ265", "MCQ268A", "MCQ300A", "MCQ300B", "MCQ300C", "MCQ310", "MCQ320", "MCD330", "MCQ340", "MCQ350"]

# Make sure SEQN is included in our columns

# Load your data
data = pd.read_csv('merged_data.csv')

column_list = data.columns.tolist()

counter = 0;

while(column_list[len(column_list) - 1] != 'LBDE74LC'):
    if column_list[-1][:3] == 'MCQ':
        
        if(column_list[len(column_list) - 1] not in names):
            names.append(column_list[len(column_list) - 1])
        
        
        counter += 1
    column_list.pop()


names_with_seqn = ['SEQN'] + names

# Replace missing values (NaN) with another placeholder, if needed
data = data.apply(lambda x: x.replace(np.nan, None) if x.dtype == 'object' else x)

# Show the cleaned data
print(data)

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot of IgE levels vs Age
sns.scatterplot(data=data, x='MCQ010', y='LBXIGE')



# Set plot labels and title
plt.xlabel('MCQ010')
plt.ylabel('IgE Level')
plt.title('Scatter Plot of IgE Levels vs MCQ010')

# Show the plot
# plt.show()

# Set plot labels and title






# Initialize lists to store SEQN numbers and cleaned values
seqn = [[] for _ in range(len(names))]
values_cleaned = [[] for _ in range(len(names))]

# Process data from the DataFrame, keeping track of SEQN for non-NaN values
for _, row in data[names_with_seqn].iterrows():
    current_seqn = row['SEQN']
    for i, name in enumerate(names):
        if not pd.isna(row[name]):
            seqn[i].append(current_seqn)
            values_cleaned[i].append(row[name])

# Sort values and maintain SEQN correspondence
for i, name in enumerate(names):
    if len(values_cleaned[i]) > 0:
        # Get sorted indices
        sorted_indices = sorted(range(len(values_cleaned[i])), key=lambda k: values_cleaned[i][k])
        # Apply indices to both lists
        values_cleaned[i] = [values_cleaned[i][j] for j in sorted_indices]
        seqn[i] = [seqn[i][j] for j in sorted_indices]

corresponding_seqn = [];
for i in range(0, 42000):
    corresponding_seqn.append(0);

for i in range(0, 9440):
    rowNum = i;
    seqNum = data['SEQN'].iloc[rowNum]
    corresponding_seqn[int(seqNum)] = rowNum



# Print info about each question's responses with their SEQN numbers
for i, name in enumerate(names):
    print(f"\nQuestion {name}:")
    print(f"Total valid responses: {len(values_cleaned[i])}")
    if len(values_cleaned[i]) > 0:
        print(f"First 5 pairs (SEQN, value): {list(zip(seqn[i][:5], values_cleaned[i][:5]))}")
        print(f"Last 5 pairs (SEQN, value): {list(zip(seqn[i][-5:], values_cleaned[i][-5:]))}")
        # Print the full range of SEQN numbers
        print(f"SEQN range: {min(seqn[i]):.0f} to {max(seqn[i]):.0f}")

# Initialize rangeInfo with the correct dimensions
rangeInfo = []

# Print debug information
print(f"Column list length: {len(column_list)}")
print(f"Values cleaned length: {len(values_cleaned)}")

# Print some debug info about the first few entries
for i in range(min(5, len(values_cleaned))):
    print(f"\nQuestion {names[i]}:")
    print(f"SEQN values: {seqn[i][:5]}")
    print(f"Values: {values_cleaned[i][:5]}")

# Now process the data

for a in range(len(column_list)):
    rangeInfo.append([])
    for b in range(len(values_cleaned)):
        rangeInfo[a].append([])
        for c in range(len(seqn[b])):
            seqnNum = int(seqn[b][c])
            rowNum = corresponding_seqn[seqnNum]
            rangeInfo[a][b].append(data[column_list[a]].iloc[rowNum])


'''
for b in range(len(values_cleaned)):
    for c in range(len(seqn[b])):
        seqnNum = int(seqn[b][c])
        rowNum = corresponding_seqn[seqnNum]
        if 0 <= rowNum < len(data):
            for a in range(len(column_list)):
                value = data[column_list[a]].iloc[rowNum]
                if len(rangeInfo[b][c]) <= a:
                    rangeInfo[b][c].append(value)
                else:
                    rangeInfo[b][c][a] = value


'''

print(values_cleaned)
print(seqn)

for i in range(len(values_cleaned)):
    print(len(values_cleaned[i]) - len(seqn[i]))
    if len(values_cleaned[i]) > 0:
        row = data[data['SEQN'] == seqn[i][0]]
        mcq = names[i]
        print(values_cleaned[i][0] - row[mcq].iloc[0])





for a in range(len(column_list)):
    for b in range(len(values_cleaned)):
        if len(values_cleaned[b]) == 0:
            continue
        expected = rangeInfo[a][b][0]
        actual = data[column_list[a]].iloc[corresponding_seqn[int(seqn[b][0])]]
        if math.isnan(expected):
            continue
           
        result = expected == actual
        print(result)
'''
print(column_list)
print(rangeInfo)
'''
count = 0;

for a in range(0, 9440):
    if data['LBDID2LC'].iloc[a] == 5.397605346934028e-79:
        count += 1

print(len(rangeInfo[7]) - len(values_cleaned))


nameScores = []

usedIGE = ['LBDIF2LC', 'LBDIG5LC', 'LBDF13LC', 'LBDID1LC', 'LBDID2LC']
for b in range (len(names)):
    score = 0
    yesNum = 0
    noNum = 0



    for d in range(0, 9440):
        if data[names[b]].iloc[d] == 1:
            yesNum += 1
        elif data[names[b]].iloc[d] == 2:
            noNum += 1

    for a in range(len(column_list)):
        
        
        for c in range(len(rangeInfo[a][b])):
            if(rangeInfo[a][b][c] == 1 and (column_list[a] in usedIGE)):
                if(values_cleaned[b][c] == 1):
                    if(yesNum > 0):
                        score += 1/yesNum
                elif(values_cleaned[b][c] == 2):
                    if(noNum > 0):
                        score -= 1/noNum

    score = abs(score)
    nameScores.append(score)

print(nameScores)
print(names)

allergyScores = []

yesAmount = 0
noAmount = 0

for a in range(len(values_cleaned)):
    for b in range(len(values_cleaned[a])):
        if(values_cleaned[a][b] == 1):
            yesAmount += 1
        elif(values_cleaned[a][b] == 2):
            noAmount += 1

for a in range(len(column_list)):
    score = 0;
    presNum = 0

    for d in range(0, 9440):
        if data[column_list[a]].iloc[d] == 2:
            presNum += 1

    for b in range(len(names)):
        for c in range(len(rangeInfo[a][b])):
            if(rangeInfo[a][b][c] == 2):
                if(values_cleaned[b][c] == 1):
                    score += 1/yesAmount;
                elif(values_cleaned[b][c] == 2):
                    score -= 1/noAmount;

    if(presNum > 0):
        score = score/presNum
    else:
        score = 0
    score = abs(score)
    allergyScores.append(score)

print(allergyScores)
print(1.0 == 1)

for a in range(len(names)):
    print(names[a] + ': ' + str(nameScores[a]))

for a in range(len(column_list)):
    print(column_list[a] + ': ' + str(allergyScores[a]))

# Populate rangeInfo using direct index access
'''
for i in range(len(seqn)):
    for a in range(len(seqn[i])):
        seqnNum = int(seqn[i][a])
        rowNum = seqnNum - 31128
        if 0 <= rowNum < len(data):
            for b in range(len(column_list)):
                # Use .iloc with the column name directly
                value = data.iloc[rowNum][column_list[b]]
                rangeInfo[i][a].append(value)

# Verify the first SEQN number for each MCQ across all columns
for b in range(len(values_cleaned)):
    if len(seqn[b]) > 0:  # Check if there are any SEQNs for this MCQ
        seqnNum = int(seqn[b][0])
        rowNum = seqnNum - 31128
        if 0 <= rowNum < len(data):
            print(f"\nChecking MCQ {names[b]}")
            print(f"SEQN: {seqnNum}")
            print(f"rowNum: {rowNum}")
            for a in range(len(column_list)):
                expected_value = data[column_list[a]].iloc[rowNum]
                actual_value = rangeInfo[b][0][a]
                print(f"Column {column_list[a]}: Expected={expected_value}, Actual={actual_value}, Match={actual_value == expected_value}")



scores = [];

for a in range (len(values_cleaned)):
    score = 0;
    for b in range(len(rangeInfo)):
        for c in range(len(rangeInfo[b][a])):
            if(c < 5/10 * len(rangeInfo[b][a])):
                if(math.isnan(rangeInfo[b][a][c])):
                    continue;
                if(int(rangeInfo[b][a][c]) == 1):
                    score += 1;
            else:
                if(math.isnan(rangeInfo[b][a][c])):
                    continue;
                if(int(rangeInfo[b][a][c]) == 1):
                    score -= 1;
    print(score);
    scores.append(score);               
'''