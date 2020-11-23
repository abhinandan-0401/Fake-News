from model import *
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess')
    parser.add_argument('--retrain')
    parser.add_argument('--predict')
    parser.add_argument('--prevId', default='')
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--outpath')
    args = parser.parse_args()

    if int(args.preprocess) == 1:
        train = pd.read_csv(str(args.train))
        train['title'] = train['title'].fillna("")
        train['text'] = train['text'].fillna("")
        train['News'] = (train['title'].map(str) +' '+ train['text']).apply(lambda row: row.strip())

        sentences, token_lists, idx_in = preprocess(train['News'].values)

        s = {}
        t = {}
        for i in range(len(idx_in)):
            s[idx_in[i]] = sentences[i] 
            t[idx_in[i]] = token_lists[i] 
        
        train['Sentences'] = train['id'].map(s)
        train['Tokens'] = train['id'].map(t)

        train.to_csv(str(args.outpath) + 'processed_train.csv', index=False)

        test = pd.read_csv(str(args.test))
        test['title'] = test['title'].fillna("")
        test['text'] = test['text'].fillna("")
        test['Review'] = (test['title'].map(str) +' '+ test['text']).apply(lambda row: row.strip())

        sentences, token_lists, idx_in = preprocess(test['Review'].values)
        
        s = {}
        t = {}
        for i in range(len(idx_in)):
            s[idx_in[i]] = sentences[i] 
            t[idx_in[i]] = token_lists[i] 
        
        test['Sentences'] = test['id'].map(s)
        test['Tokens'] = test['id'].map(t)

        test.to_csv(str(args.outpath) + 'processed_test.csv', index=False)
    else:
        if int(args.predict) != 1:
            if int(args.retrain) == 0:
                train = pd.read_csv(str(args.train))
                train['Sentences'] = train['Sentences'].fillna("")
                
                test = pd.read_csv(str(args.test))
                test['Sentences'] = test['Sentences'].fillna("")

                # Define the bert classification model object
                sc = Sequence_Classification(2, 32, 256)
                sc.model_dir = "//tomtom/models/"

                sc.fit(train['Sentences'].values, train['label'].values)
                sc.model = None

                # # save the model
                with open(sc.model_dir + "{}.file".format(sc.id), "wb") as f:
                    pickle.dump(sc, f, pickle.HIGHEST_PROTOCOL)

                # predict on test data
                labels = sc.predict(test['Sentences'].values)
                sub = pd.concat([test['id'],pd.DataFrame(labels, columns=['label'])], axis = 1)
                sub.to_csv(str(args.outpath) + 'submit.csv', index=False)
            else:
                train = pd.read_csv(str(args.train))
                train['Sentences'] = train['Sentences'].fillna("")
                
                test = pd.read_csv(str(args.test))
                test['Sentences'] = test['Sentences'].fillna("")

                # Define the bert classification model object
                sc = Sequence_Classification(2, 32, 256)
                sc.model_dir = "/tomtom/models/"
                sc.id = str(args.prevId)

                sc.re_fit(train['Sentences'].values, train['label'].values)
                sc.model = None

                # # save the model
                with open(sc.model_dir + "{}.file".format(sc.id), "wb") as f:
                    pickle.dump(sc, f, pickle.HIGHEST_PROTOCOL)

                # predict on test data
                labels = sc.predict(test['Sentences'].values)
                sub = pd.concat([test['id'],pd.DataFrame(labels, columns=['label'])], axis = 1)
                sub.to_csv(str(args.outpath) + 'submit.csv', index=False)
        else:
            print(" Prediction Mechanism Started... ")
            test = pd.read_csv(str(args.test))
            test['Sentences'] = test['Sentences'].fillna("")

            # # save the model            
            with open("/tomtom/models/" + "{}.file".format(str(args.prevId)), "rb") as f:
                sc = pickle.load(f)

            # predict on test data
            labels = sc.predict(test['Sentences'].values)
            sub = pd.concat([test['id'],pd.DataFrame(labels, columns=['label'])], axis = 1)
            # print(sub['label'].value_counts())
            sub.to_csv(str(args.outpath) + 'submit.csv', index=False)
            print(" Prediction Mechanism Ended... ")