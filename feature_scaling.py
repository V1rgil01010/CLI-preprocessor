import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from data_description import DataDescription
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
class FeatureScaling:
    
    bold_start = "\033[1m"
    bold_end = "\033[0;0m"
    

    tasks = [
        "\n1. Perform PCA on the whole Dataset",
        "2. Show the Dataset"
    ]

    # All the Tasks associated with this class.
    tasks = [
        "\n1. Perform Normalization(MinMax Scaler)",
        "2. Perform Standardization(Standard Scaler)",
        "3. Perform MaxAbs Scaler",
        "4. Perform Robust Scaler",
        "5. Remove outliers",
        "6. removeoutlierswholeds",
        "7. LDA",
        "8. pca",
        "9. Singular Value Decomposition (SVD)",
        "10. Smooth by bin means",
        "11. Smooth by bin median",
        "12. Show the Dataset"
    ]
    
    tasks_normalization = [
        "\n1. Normalize a specific Column",
        "2. Normalize the whole Dataset",
        "3. Show the Dataset"
    ]

    tasks_standardization = [
        "\n1. Standardize a specific Column",
        "2. Standardize the whole Dataset",
        "3. Show the Dataset"
    ]
    tasks_maxabs = [
        "\n1. apply MaxAbs Scaler on a specific Column",
        "2. apply MaxAbs Scaler on the whole Dataset",
        "3. Show the Dataset"
    ]
    tasks_robust_scaling = [
        "\n1. apply Robust Scaler on a specific Column",
        "2. apply Robust Scaler on the whole Dataset",
        "3. Show the Dataset"
    ]
    tasks_pca = [
        "\n1. Perform PCA on the whole Dataset",
        "2. Show the Dataset"
    ]

    tasks_smoothbybinmeans = [
        "\n1. Perform smoothbybinmeans on the single column",
        "2.Perform smoothbybinmeans on the whole Dataset ",
        "3.Show the Dataset"
    ]
    tasks_smoothingbymedian = [
        "\n1. Perform smoothbybinmedian on the single column",
        "2.Perform smoothbybinmedian on the whole Dataset ",
        "3.Show the Dataset"
    ]
    
    def __init__(self, data):
        self.data = data
    
    # Performs Normalization on specific column or on whole dataset.
    def normalization(self):
        while(1):
            print("\nTasks (Normalization)\U0001F447")
            for task in self.tasks_normalization:
                print(task)

            while(1):
                try:
                    choice = int(input(("\n\nWhat you want to do? (Press -1 to go back)  ")))
                except ValueError:
                    print("Integer Value required. Try again.....\U0001F974")
                    continue
                break
    
            if choice == -1:
                break
            
            # Performs normalization on the columns provided.
            elif choice == 1:
                print(self.data.dtypes)
                columns = input("Enter all the column"+ self.bold_start + "(s)" + self.bold_end + "you want to normalize (Press -1 to go back)  ").lower()
                if columns == "-1":
                    break
                for column in columns.split(" "):
                    # This is the basic approach to perform MinMax Scaler on a set of data.
                    try:
                        minValue = self.data[column].min()
                        maxValue = self.data[column].max()
                        self.data[column] = (self.data[column] - minValue)/(maxValue - minValue)
                    except:
                        print("\nNot possible....\U0001F636")
                print("Done....\U0001F601")

            # Performs normalization on whole dataset.
            elif choice == 2:
                try:
                    self.data = pd.DataFrame(MinMaxScaler().fit_transform(self.data))
                    print("Done.......\U0001F601")

                except:
                    print("\nString Columns are present. So, " + self.bold_start + "NOT" + self.bold_end + " possible.\U0001F636\nYou can try the first option though.")
                
            elif choice==3:
                DataDescription.showDataset(self)

            else:
                print("\nYou pressed the wrong key!! Try again..\U0001F974")

        return

    # Function to perform standardization on specific column(s) or on whole dataset.
    def standardization(self):
        while(1):
            print("\nTasks (Standardization)\U0001F447")
            for task in self.tasks_standardization:
                print(task)

            while(1):
                try:
                    choice = int(input(("\n\nWhat you want to do? (Press -1 to go back)  ")))
                except ValueError:
                    print("Integer Value required. Try again.....")
                    continue
                break

            if choice == -1:
                break
            
            # This is the basic approach to perform Standard Scaler on a set of data.
            elif choice == 1:
                print(self.data.dtypes)
                columns = input("Enter all the column"+ self.bold_start + "(s)" + self.bold_end + "you want to normalize (Press -1 to go back)  ").lower()
                if columns == "-1":
                    break
                for column in columns.split(" "):
                    try:
                        mean = self.data[column].mean()
                        standard_deviation = self.data[column].std()
                        self.data[column] = (self.data[column] - mean)/(standard_deviation)
                    except:
                        print("\nNot possible....\U0001F636")
                print("Done....\U0001F601")
                    
            # Performing standard scaler on whole dataset.
            elif choice == 2:
                try:
                    self.data = pd.DataFrame(StandardScaler().fit_transform(self.data))
                    print("Done.......\U0001F601")
                except:
                    print("\nString Columns are present. So, " + self.bold_start + "NOT" + self.bold_end + " possible.\U0001F636\nYou can try the first option though.")
                break

            elif choice==3:
                DataDescription.showDataset(self)

            else:
                print("\nYou pressed the wrong key!! Try again..\U0001F974")

        return
    def maxabs_scaler(self):
        while(1):
            print("\nTasks (MaxAbs Scaler)\U0001F447")
            for task in self.tasks_maxabs:
                print(task)

            while(1):
                try:
                    choice = int(input(("\n\nWhat you want to do? (Press -1 to go back)  ")))
                except ValueError:
                    print("Integer Value required. Try again.....\U0001F974")
                    continue
                break

            if choice == -1:
                break

        # Performs MaxAbs Scaler on the columns provided.
            elif choice == 1:
                print(self.data.dtypes)
                columns = input("Enter all the column" + self.bold_start + "(s)" + self.bold_end + "you want to scale (Press -1 to go back)  ").lower()
                if columns == "-1":
                    break
                for column in columns.split(" "):
                # This is the basic approach to perform MaxAbs Scaler on a set of data.
                    try:
                        maxabs_scaler = MaxAbsScaler()
                        self.data[column] = maxabs_scaler.fit_transform(self.data[column].values.reshape(-1,1))
                    except:
                        print("\nNot possible....\U0001F636")
                print("Done....\U0001F601")

        # Performs MaxAbs Scaler on whole dataset.
            elif choice == 2:
                try:
                    self.data = pd.DataFrame(MaxAbsScaler().fit_transform(self.data))
                    print("Done.......\U0001F601")
                except:
                    print("\nString Columns are present. So, " + self.bold_start + "NOT" + self.bold_end + " possible.\U0001F636\nYou can try the first option though.")
                    break

            elif choice==3:
                DataDescription.showDataset(self)

            else:
                print("\nYou pressed the wrong key!! Try again..\U0001F974")

        return    
    def robust_scaling(self):
        while True:
            print("\nTasks (Robust Scaler)\U0001F447")
            for task in self.tasks_robust_scaling:
                print(task)

            while True:
                try:
                    choice = int(input("\nWhat you want to do? (Press -1 to go back) "))
                except ValueError:
                    print("Please enter a valid integer value.")
                    continue
                break

            if choice == -1:
                break

        # Performs Robust Scaler on the columns provided
            elif choice == 1:
                print(self.data.dtypes)
                columns = input("Enter all the column(s) you want to apply Robust Scaler on (Press -1 to go back) ").lower()
                if columns == "-1":
                    break
                for column in columns.split(" "):
                    try:
                        scaler = RobustScaler()
                        self.data[column] = scaler.fit_transform(self.data[[column]])
                    except:
                        print("\nNot possible....\U0001F636")
                print("Done....\U0001F601")

        # Performs Robust Scaler on whole dataset
            elif choice == 2:
                try:
                    scaler = RobustScaler()
                    self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
                    print("Done.......\U0001F601")
                except:
                    print("\nString Columns are present. So, " + self.bold_start + "NOT" + self.bold_end + " possible.\U0001F636\nYou can try the first option though.")

            elif choice == 3:
                DataDescription.showDataset(self)

            else:
                print("\nInvalid choice! Please enter a valid choice.")
            
        return
    def removeoutliers(self):
        # Get binary enco
        int_columns=self.data.select_dtypes(include='integer').columns.tolist()
        while(1):
            column_name99 = input("\nWhich column would you like to remove outliers from? (Press -1 to go back)  ").lower()
            if column_name99 == "-1":
                break
            if column_name99 in int_columns:
                Threshold = int(input(("\n\nWhat is the threshold? (Press -1 to go back)  ")))
                z_score = (self.data[column_name99] - self.data[column_name99].mean()) / self.data[column_name99].std()
                self.data = self.data[abs(z_score) <= Threshold]
                print(f"Outliers removed from column '{column_name99}' using Z-score method.......\U0001F601")
                choice = input("Are there more ouliers removed from the columns ? (y/n)  ")
                if choice == "y" or choice == "Y":
                    continue
                else:
                    DataDescription.showDataset(self)
                    break
            else:
                print("Wrong Column Name. Try Again...\U0001F975")


        
    def removeoutlierswholeds(self):
        # Get binary e
        Threshold = int(input(("\n\nWhat is the threshold? (Press -1 to go back)  ")))
        z_scores = np.abs((self.data - self.data.mean()) / self.data.std())
                # Replace any value with a z-score greater than 3 with NaN
        self.data[z_scores > Threshold] = np.nan
        # Drop any row with at least one NaN value
        self.data.dropna(inplace=True)
        print(f"Outliers removed from the dataset Z-score method.......\U0001F601")
        
        DataDescription.showDataset(self)

            
    def lda(self):
        while True:
            column_name = input("\nEnter the name of the column you want to use as the target: ")
            if column_name == "-1":
                break
            if column_name not in self.data.columns:
                print("Target column not found in dataset")
                return
            else:
                target = self.data[column_name]
                X = self.data.drop([column_name], axis=1)

                try:
                    lda = LinearDiscriminantAnalysis()
                    X_lda = lda.fit_transform(X, target)
                    self.data = pd.concat([pd.DataFrame(X_lda), target], axis=1)
                    print("LDA performed successfully!\n")
                    return X_lda
                except:
                    print("LDA could not be performed. Please try again.")
                    return None


        
      
        print("LDA performed successfully!")

    def pca(self):
        while True:
            print("\nTasks (pca)\U0001F447")
            for task in self.tasks_pca:
                print(task)

            while True:
                try:
                    choice = int(input("\nWhat you want to do? (Press -1 to go back) "))
                except ValueError:
                    print("Please enter a valid integer value.")
                    continue
                break

            if choice == -1:
                break

            # Performs PCA on whole dataset
            elif choice == 1:
                try:
                    n_components = int(input("Enter the number of principal components to keep: "))
                    pca = PCA(n_components=n_components)
                    principal_components = pca.fit_transform(self.data)
                    principal_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
                    self.data = pd.concat([principal_df, self.data.reset_index(drop=True)], axis=1)
                    print(f"{n_components} principal components kept and added to the dataset.......\U0001F601")
                except:
                    print("\nSomething went wrong. Please try again.\U0001F636")

            elif choice == 2:
                DataDescription.showDataset(self)

            else:
                print("\nInvalid choice! Please enter a valid choice.")

        return
    def svd(self):
        n_components=int(input(("\n\n Number of Components (Press -1 to go back)  ")))
        try:
            svd = TruncatedSVD(n_components=n_components)
            X_svd = svd.fit_transform(self.data)
            print("Singular Value Decomposition is done....\U0001F601")
            print("SVD performed successfully!\n")
            
            return X_svd
        except:
            print("SVD could not be performed. Please try again.")
            return None
    def smoothing_by_bin_means(self):
        while True:
            print("\nTasks (Smoothing by Bin Means) \U0001F447")
            for task in self.tasks_smoothbybinmeans:
                print(task)

            while True:
                try:
                    choice = int(input("\nWhat do you want to do? (Press -1 to go back) "))
                except ValueError:
                    print("Please enter a valid integer value.")
                    continue
                break

            if choice == -1:
                break

            # Performs smoothing by bin means on a single column
            elif choice == 1:
                column = input("Enter the column name to perform smoothing: ")
                bins = int(input("Enter the number of bins: "))

                try:
                    self.data[column] = self.data.groupby(pd.cut(self.data[column], bins=bins))[column].transform('mean')
                    print("Smoothing by bin means is performed on the column....\U0001F601")
                except:
                    print("\nSomething went wrong. Please try again.\U0001F636")

            # Performs smoothing by bin means on the whole dataset
            elif choice == 2:
                bins = int(input("Enter the number of bins: "))

                try:
                    self.data = self.data.apply(lambda x: self.data.groupby(pd.cut(x, bins=bins)).transform('mean') if np.issubdtype(x.dtype, np.number) else x)
                    print("Smoothing by bin means is performed on the whole dataset....\U0001F601")
                except:
                    print("\nSomething went wrong. Please try again.\U0001F636")

            # Displays the dataset
            elif choice == 3:
                DataDescription.showDataset(self)

            else:
                print("\nInvalid choice! Please enter a valid choice.")

        return  
   

    def smoothing_by_bin_median(self):
        while True:
            print("\nTasks (Smoothing by Bin Median) \U0001F447")
            for task in self.tasks_smoothingbymedian:
                print(task)

            while True:
                try:
                    choice = int(input("\nWhat do you want to do? (Press -1 to go back) "))
                except ValueError:
                    print("Please enter a valid integer value.")
                    continue
                break

            if choice == -1:
                break

            # Performs smoothing by bin median on a single column
            elif choice == 1:
                column = input("Enter the column name to perform smoothing: ")
                bins = int(input("Enter the number of bins: "))

                try:
                    self.data[column] = self.data.groupby(pd.cut(self.data[column], bins=bins))[column].transform('median')
                    print("Smoothing by bin median is performed on the column....\U0001F601")
                except:
                    print("\nSomething went wrong. Please try again.\U0001F636")

            # Performs smoothing by bin median on the whole dataset
            elif choice == 2:
                bins = int(input("Enter the number of bins: "))

                try:
                    self.data = self.data.apply(lambda x: self.data.groupby(pd.cut(x, bins=bins)).transform('median') if np.issubdtype(x.dtype, np.number) else x)
                    print("Smoothing by bin median is performed on the whole dataset....\U0001F601")
                except:
                    print("\nSomething went wrong. Please try again.\U0001F636")

            # Displays the dataset
            elif choice == 3:
                DataDescription.showDataset(self)

            else:
                print("\nInvalid choice! Please enter a valid choice.")

        return

    
    # main function of the FeatureScaling Class.
    def scaling(self):
        while(1):
            print("\nTasks (Feature Scaling)\U0001F447")
            for task in self.tasks:
                print(task)
            
            while(1):
                try:
                    choice = int(input(("\n\nWhat you want to do? (Press -1 to go back)  ")))
                except ValueError:
                    print("Integer Value required. Try again.....\U0001F974")
                    continue
                break
            if choice == -1:
                break
            
            elif choice == 1:
                self.normalization()

            elif choice == 2:
                self.standardization()
            elif choice == 3:
                self.maxabs_scaler()
            elif choice == 4:
                self.robust_scaling()        
            elif choice == 5:
                self.removeoutliers()
            elif choice == 6:
                self.removeoutlierswholeds()        

            elif choice == 7:
                self.lda()
            elif choice == 8:
                self.pca()

            elif choice == 9:
                self.svd() 
            elif choice == 10:
                self.smoothing_by_bin_means()   
            elif choice == 11:
                self.smoothing_by_bin_median()              

            elif choice == 12:
                DataDescription.showDataset(self)    
            
            else:
                print("\nWrong Integer value!! Try again..\U0001F974")
        # Returns all the changes on the DataFrame.
        return self.data