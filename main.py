import src.category_encoding as data_cleaning
import src.data_expansion as data_expansion
import preprocess_data
import baseline_lr_expanded
import svm_expanded

def main():
    #preprocess_data.main()
    baseline_lr_expanded.run()
    svm_expanded.run()

if __name__ == '__main__':
    main()