import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from data_description import DataDescription
import category_encoders as ce
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
wn = nltk.stem.WordNetLemmatizer()


class Categorical:
    # The Task associated with this class.
    tasks = [
        '\n1. Show Categorical Columns',
        '2. Performing One Hot encoding',
        '3. Performing Hash encoding',
        '4. Performing targetEncoding',
        '5. Performing Leaveoneoutencoding',
        '6. Performing binaryEncoding',
        '7. Performing labelEncoding',
        '8. Performing grayEncoding',
        '9. Performing catboostEncoding',
        '10. Performing PolynomialEncoding',
        '11. Performing backwarddifferenceEncoding',
        '12. Performing Text Preprocessing',
        '13. Show the Dataset'
    ]
    tasks_text = [
        '\n1. Remove Punctuation marks',
        '2. Remove Stop Words',
        '3. Convert to Lowercase',
        '4. Lemmatization',
        '5. Stemming',
        '6. Tokenization',
        '7. Show the Dataset'
    ]
    def __init__(self, data):
        self.data = data
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # function to show all the categorical columns and number of unique values in them.
    def categoricalColumn(self):
        print('\n{0: <20}'.format("Categorical Column") + '{0: <5}'.format("Unique Values"))
        # select_dtypes selects the columns with object datatype(which could be further categorize)
        for column in self.data.select_dtypes(include="object"):
            print('{0: <20}'.format(column) + '{0: <5}'.format(self.data[column].nunique()))

    # function to encode any particular column
    def onehotencoding(self):
        categorical_columns = self.data.select_dtypes(include="object")
        while(1):
            column = input("\nWhich column would you like to one hot encode?(Press -1 to go back)  ").lower()
            if column == "-1":
                break
            # The encoding function is only for categorical columns.
            if column in categorical_columns:
                self.data = pd.get_dummies(data=self.data, cols = [column])
                print("Encoding is done.......\U0001F601")
                
                choice = input("Are there more columns to be encoded?(y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F974")
                
    def hashingencoding(self):
        categorical_columns = self.data.select_dtypes(include="object")
        while(1):
            column_name4= input("\nWhich column would you like to hashing encode? (Press -1 to go back)  ").lower()
            if column_name4 == "-1":
                break
            if column_name4 in categorical_columns:
                encoder4 = ce.HashingEncoder(cols=[column_name4])
                encoded_data = encoder4.fit_transform(self.data)
        # update the original data with the encoded data
                self.data = pd.concat([self.data, encoded_data], axis=1)
                print("Hashing encoding is done.......\U0001F601")
                
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")
    def target_encode_column(self):
        categorical_columns = self.data.select_dtypes(include='object')
        while(1):
            column_name1 = input("Enter the name of the column to be target encoded: ")
            if column_name1 == "-1":
                break
            if column_name1 in categorical_columns:
                target_column = input("Enter the name of the target column: ")
                if target_column not in self.data.columns:
                   print("Target column not found in dataset")
                   return
                 
                le = LabelEncoder()
                self.data[target_column] = le.fit_transform(self.data[target_column])
                encoder = ce.TargetEncoder(cols=[column_name1], smoothing=0.3)
                self.data[column_name1] = encoder.fit_transform(self.data[column_name1],self.data[target_column])
                print(f"Target encoding  is done.......\U0001F601")
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")   
    def Leaveoneoutencoder(self):
        categorical_columns = self.data.select_dtypes(include='object')
        while(1):
            column_name2 = input("Enter the name of the column to be leaveoneout encoded: ")
            if column_name2 == "-1":
                break
            if column_name2 in categorical_columns:
                target_column1 = input("Enter the name of the target column: ")
                if target_column1 not in self.data.columns:
                   print("Target column not found in dataset")
                   return
                 
                le = LabelEncoder()
                self.data[target_column1] = le.fit_transform(self.data[target_column1])
                encoder1 = ce.LeaveOneOutEncoder(cols=[column_name2])
                self.data[column_name2] = encoder1.fit_transform(self.data[column_name2],self.data[target_column1])
                print(f"Leaveoneout encoding  is done.......\U0001F601")
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")                                 
    def binaryencoding(self):
        # Get binary encodings for all categorical columns
        categorical_columns = self.data.select_dtypes(include='object')
        while(1):
            column_name = input("\nWhich column would you like to binary encode? (Press -1 to go back)  ").lower()
            if column_name == "-1":
                break
            if column_name in categorical_columns:
                encoder = ce.BinaryEncoder(cols=[column_name])
                self.data = encoder.fit_transform(self.data)
                print("Binary encoding is done.......\U0001F601")
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")
         
    def labelencoding(self):
        # Get binary encodings for all categorical columns
        categorical_columns = self.data.select_dtypes(include='object')
        while(1):
            column_name3 = input("\nWhich column would you like to label encode? (Press -1 to go back)  ").lower()
            if column_name3 == "-1":
                break
            if column_name3 in categorical_columns:
                
                encoder2 = LabelEncoder()
                self.data[column_name3] = encoder2.fit_transform(self.data[column_name3])

                print("Label encoding is done.......\U0001F601")
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")
    def grayencoding(self):
        # Get binary encodings for all categorical columns
        categorical_columns = self.data.select_dtypes(include='object')
        while(1):
            column_name5 = input("\nWhich column would you like to gray encode? (Press -1 to go back)  ").lower()
            if column_name5 == "-1":
                break
            if column_name5 in categorical_columns:
                encoder6 = ce.GrayEncoder(cols=[column_name5])
                self.data = encoder6.fit_transform(self.data)
                print("gray encoding is done.......\U0001F601")
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")  

    def catboostencoder(self):
        categorical_columns = self.data.select_dtypes(include='object')
        while(1):
            column_name8 = input("Enter the name of the column to be catboost encoded: ")
            if column_name8 == "-1":
                break
            if column_name8 in categorical_columns:
                target_column2 = input("Enter the name of the target column: ")
                if target_column2 not in self.data.columns:
                   print("Target column not found in dataset")
                   return
                 
                le = LabelEncoder()
                self.data[target_column2] = le.fit_transform(self.data[target_column2])
                encoder5 = ce.CatBoostEncoder(cols=[column_name8])
                self.data[column_name8] = encoder5.fit_transform(self.data[column_name8],self.data[target_column2])
                print(f"catboost encoding  is done.......\U0001F601")
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")     
    def polynomialencoding(self):
        # Get binary encodings for all categorical columns
        categorical_columns = self.data.select_dtypes(include='object')
        while(1):
            column_name57 = input("\nWhich column would you like to polynomial encode? (Press -1 to go back)  ").lower()
            if column_name57 == "-1":
                break
            if column_name57 in categorical_columns:
                encoder67 = ce.PolynomialEncoder(cols=[column_name57])
                self.data = encoder67.fit_transform(self.data)
                print("polynomial encoding is done.......\U0001F601")
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")  
    
    def backwarddifferenceencoding(self):
        # Get binary encodings for all categorical columns
        categorical_columns = self.data.select_dtypes(include='object')
        while(1):
            column_name512 = input("\nWhich column would you like to backwarddifference encode? (Press -1 to go back)  ").lower()
            if column_name512 == "-1":
                break
            if column_name512 in categorical_columns:
                encoder612 = ce.BackwardDifferenceEncoder(cols=[column_name512])
                self.data = encoder612.fit_transform(self.data)
                print("backwarddifference encoding is done.......\U0001F601")
                choice = input("Are there more columns to be encoded? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    self.categoricalColumn()
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")
   

    def preprocess_text(self):
        while(1):
            print("\nTasks (Text Preprocessing)\U0001F447")
            for task in self.tasks_text:
                print(task)
            text_column = input("\nWhich column would you like to preprocess? (Press -1 to go back)  ").lower()
            while(1):
                try:
                    choice = int(input(("\n\nWhat you want to do? (Press -1 to go back)  ")))
                except ValueError:
                    print("Integer Value required. Try again.....\U0001F974")
                    continue
                break
    
            if choice == -1:
                break

        

            
            #choice1 = input("\nWhat would you like to do with this text column?\n" + "\n".join(Text.tasks) + "\n(Press -1 to go back) ")
            elif choice == 1:
                self.data[text_column] = self.data[text_column].str.replace('[^\w\s]','')
                print("Punctuation marks have been removed.......\U0001F601")
            elif choice == 2:
                nltk.download('stopwords')
                stop_words = set(stopwords.words('english'))
                self.data[text_column] = self.data[text_column].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
                print("Stop words have been removed.......\U0001F601")
            elif choice == 3:
                self.data[text_column] = self.data[text_column].str.lower()
                print("Text has been converted to lowercase.......\U0001F601")
            elif choice == 4:
                nltk.download('wordnet')
                wn = nltk.stem.WordNetLemmatizer()
                self.data[text_column] = self.data[text_column].apply(lambda x: ' '.join([wn.lemmatize(word) for word in x.split()]))
                print("Text has been lemmatized.......\U0001F601")
            elif choice == 5:
                ps = nltk.stem.PorterStemmer()
                self.data[text_column + '_stemmed'] = self.data[text_column].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
                print("Text has been stemmed.......\U0001F601")
            elif choice ==6:
                self.data[text_column + '_tokens'] = self.data[text_column].apply(lambda x: word_tokenize(x))
                print("Text has been tokenized.......\U0001F601")
            elif choice == 7:
                print(self.data)
            elif choice == '-1':
                break
            else:
                print("Invalid choice. Try again.\U0001F974")                                                                 

    # The main function of the Categorical class.
    def categoricalMain(self):
        while(1):
            print("\nTasks\U0001F447")
            for task in self.tasks:
                print(task)

            while(1):
                try:
                    choice = int(input(("\n\nWhat you want to do? (Press -1 to go back)  ")))
                except ValueError:
                    print("Integer Value required. Try again...\U0001F974")
                    continue
                break

            if choice == -1:
                break
            
            elif choice == 1:
                self.categoricalColumn()

            elif choice == 2:
                self.categoricalColumn()
                self.onehotencoding()
            elif choice == 3:
                self.categoricalColumn()
                self.hashingencoding()
            elif choice == 4:
                self.categoricalColumn()
                self.target_encode_column()
            elif choice == 5:
                self.categoricalColumn()
                self.Leaveoneoutencoder()           
            elif choice == 6:
                self.categoricalColumn()
                self.binaryencoding()
            elif choice == 7:
                self.categoricalColumn()
                self.labelencoding() 
            elif choice == 8:
                self.categoricalColumn()
                self.grayencoding()   
            elif choice == 9:
                self.categoricalColumn()
                self.catboostencoder()
            elif choice == 10:
                self.categoricalColumn()
                self.polynomialencoding() 
            elif choice == 11:
                self.categoricalColumn()
                self.backwarddifferenceencoding()  
            elif choice == 12:
                self.categoricalColumn()
                self.preprocess_text()                 
            elif choice == 13:
                DataDescription.showDataset(self)

            else:
                print("\nWrong Integer value!! Try again..\U0001F974")
        # return the data after modifying
        return self.data
