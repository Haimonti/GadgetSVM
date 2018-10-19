#Run on data and split it into n files


import os

if __name__ == "__main__":
    from sklearn.datasets import load_svmlight_file, dump_svmlight_file
    import sys
    from sklearn.model_selection import train_test_split
    

    datafolder = sys.argv[1]
    file = sys.argv[2]
    print(datafolder)
    data = load_svmlight_file(os.path.join(datafolder, file))
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.33, random_state=42)
    dump_svmlight_file(X_train, y_train, os.path.join(datafolder, "train.dat"))
    dump_svmlight_file(X_test, y_test, os.path.join(datafolder, "test.dat"))
    print("Done")