def confusion_matrix(y_true, y_pred):
    matrix=[
        [0,0],
        [0,0]
    ]

    for actual,predicted in zip(y_true,y_pred):
        if actual==1 and predicted==1:
            matrix[0][0]+=1
        elif actual==1 and predicted==-1:
            matrix[0][1]+=1
        elif actual==-1 and predicted==1:
            matrix[1][0]+=1
        elif actual==-1 and predicted==-1:
            matrix[1][1]+=1


    return matrix


def accuracy(y_true, y_pred):
    t = 0
    samples = len(y_true)

    for actual,predicted in zip(y_true,y_pred):
        if actual==1 and predicted==1 or actual==-1 and predicted==-1:
            t+=1

    return t/samples * 100