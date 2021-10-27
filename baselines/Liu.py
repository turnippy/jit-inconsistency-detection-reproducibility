from sklearn.ensemble import RandomForestClassifier

'''
Features:
    Code Features:
        Class Level:
            Changes on class attributes
            Class attribute realted 
        Method Level:
            Changes on method declaration
            Input/outpu related
        Statement Level:
            Number of statements
            Number of changed statements
            Percentage of changed statements
            Statement changes
            Refactorings
    Comment Features:
        Length of the comment
        Task comment
        Bug or version comment
        The ratio of comment lines to the class
        The ratio of comment lines to the method
        The ratio of comment lines to the code snippet
    Relationship Features:
        The similarity of code and its comment before the change
        The similarity of code and its comment after the change
        The distance of code and comments similarites
        The similarity of changed statements and the comment before the change
        The similarity of changed statements and the comment after the change
        The distance of changed statement and comment similariteis

'''


'''
n_estimatorsint = 100, selected 100 Classification And Regression Trees (CART) as the basic classifiers 
max_features = 8, set the random subset value of the features as 8
criterion = 'gini', use 'Gini' function
'''
if __name__ == "__main__":

    clf = RandomForestClassifier(n_estimatorsint=100,criterion='gini',max_features=8)
