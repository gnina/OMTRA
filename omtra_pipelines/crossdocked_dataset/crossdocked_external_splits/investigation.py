import torch
#load the file
data = torch.load('split_by_name.pt')

# Inspect the data
print("Type of Data:", type(data))

# Print keys of the dictionary
print("Keys of Data:", data.keys())

# Inspect the what each key contains
for key in list(data.keys()):
    print(f"Key: {key}, Value Type: {type(data[key])}, Length: {len(data[key])}")


# Print the first few contents of the list in training set
for i in range(5):
    print(f"Training Set Item {i}: {data['train'][i]}")
