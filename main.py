import src.category_encoding as data_cleaning
import src.data_expansion as data_expansion
import preprocess_data
import baseline_lr_expanded
import svm_expanded
import src.ROI_and_loss_dist as ROI_loss
import pandas as pd
import src.emp as emp

def main():
    #preprocess_data.main()
    # baseline_lr_expanded.run()
    # svm_expanded.run()
    # ROI_loss.run()

    emp.run()

    # pd.set_option('display.max_columns', 500)
    # x_train = pd.read_csv('train_lendingclub.txt', index_col = 0)
    # x_val = pd.read_csv('validation_lendingclub.txt', index_col = 0)
    # x_test = pd.read_csv('test_lendingclub.txt', index_col = 0)
    #
    #
    # x_train.drop(['home_ownership_NONE', 'home_ownership_OTHER', 'purpose_educational', 'addr_state_IA', 'total_pymnt', 'recoveries'], axis = 1, inplace = True)
    # x_val.drop(['total_pymnt', 'recoveries'], axis=1, inplace=True)
    # x_test.drop(['home_ownership_NONE', 'purpose_educational', 'total_pymnt', 'recoveries'], axis = 1, inplace = True)
    #
    #
    #
    # order = list(x_train)
    #
    # x_train = x_train.reindex(order, axis=1)
    # x_test = x_test.reindex(order, axis=1)
    # x_val = x_val.reindex(order, axis=1)
    #
    #
    # x_train.round(3).to_csv('train_lendingclub.txt')
    # x_test.round(3).to_csv('test_lendingclub.txt')
    # x_val.round(3).to_csv('validation_lendingclub.txt')





if __name__ == '__main__':
    main()