from sklearn import tree

def hard_voting_prediction(test_sample, forest, bagging_indices = None):
    predictions_list = list()
    for i in range(len(test_sample)):
        predictions = dict()
        predictions[0]=0
        predictions[1]=0
        predictions_list.append(predictions)
    for i in range(len(forest)):
        prediction = list()
        if bagging_indices is not None:
            test_sample_temp = test_sample[:, bagging_indices[i]]
            prediction = forest[i].predict(test_sample_temp)
        else:
            prediction = forest[i].predict(test_sample)

        for i in range(len(prediction)):
            predictions_list[i][prediction[i]]+=1
    
    final_pred = list()
    for predictions in predictions_list:
        final_pred.append(max(predictions, key= lambda x: predictions[x]))
    
    return final_pred

