import pandas as pd
from sklearn.model_selection import train_test_split

class Download:
    
    def __init__(self, data):
        self.data = data

    def download(self):
        # Split data into train and test sets
        splits = float(input(("give the split percentage  ")))
        train, test = train_test_split(self.data, test_size=splits, random_state=42)

        # Save train and test sets as separate CSV files
        filename = input("Enter the filename (without suffix) to save train and test sets: ")
        train_filename = filename + "_train.csv"
        test_filename = filename + "_test.csv"
        train.to_csv(train_filename, index=False)
        test.to_csv(test_filename, index=False)

        print("Train and test sets saved successfully!")
        print("Hurray!! It is done....\U0001F601")
        
        if input("Do you want to exit now? (y/n) ").lower() == 'y':
            print("Exiting...\U0001F44B")
            exit()
        else:
            return

