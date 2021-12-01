from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

from data import ParamDataSet

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
            Statement changes 3 * 12
            Extract Method
            Inline Method
            Rename Method
            Add Parameter
            Remove Parameter
            Inline Temp
            Encapsulate Field
            Introduce Assertion
            Replace Exception With Test
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
9 removed features:
    code: changes on class attribute
    code: class attribute related
    refactoring: extract method
    refactoring: inline method
    refactoring: encapsulate field
    refactoring: replace exception with test
    comment: ratio of comment lines to class
    comment: ratio of comment lines to the method
    comment: ratio of comment lines to the code snippet
'''


'''
    Code Features:
        Changes on method declaration
        Input/outpu related
        Number of statements
        Number of changed statements
        Percentage of changed statements
        Statement changes 3 * 12
        Rename Method
        Add Parameter
        Remove Parameter
        Inline Temp
        Introduce Assertion
    Comment Features:
        Length of the comment
        Task comment
        Bug or version comment
    Relationship Features:
        The similarity of code and its comment before the change
        The similarity of code and its comment after the change
        The distance of code and comments similarites
        The similarity of changed statements and the comment before the change
        The similarity of changed statements and the comment after the change
        The distance of changed statement and comment similariteis
'''

'''
Code Features:
    Changes on method declaration
    Input/outpu related
    Number of statements
    Number of changed statements
    Percentage of changed statements
    Statement changes 3 * 12
    Rename Method
    Add Parameter
    Remove Parameter
    Inline Temp
    Introduce Assertion
'''

class CodeFeatures:

    def ChangesOnMethod(self):
        pass

    def InputOutputRelated(self):
        pass

'''
Comment Features:
    Length of the comment
    Task comment
    Bug or version comment
'''

class CommentFeatures:

    def __init__(self,comment):
        self.comment = comment

    def LengthComment(self):
        return len(self.comment)

    def TaskComment(self):
        tasks = ['todo','fixme','xxx']
        return 1 if any([x.lower() in tasks for x in self.comment]) else 0

    def BufOrVersionComment(self):
        key_words = ['bug','fixed bug','version']
        return 1 if any([x.lower() in key_words for x in self.comment]) else 0

    def extract(self):
        return [self.LengthComment(),self.TaskComment(),self.BufOrVersionComment()]


'''
Relationship Features:
    The similarity of code and its comment before the change
    The similarity of code and its comment after the change
    The distance of code and comments similarites
    The similarity of changed statements and the comment before the change
    The similarity of changed statements and the comment after the change
    The distance of changed statement and comment similariteis
'''
from difflib import SequenceMatcher

class RelationshipFeatures:

    def __init__(self,old_code,new_code,old_comment):
        self.old_code = old_code
        self.new_code = new_code
        self.old_comment = old_comment
        self.new_comment = None

    def sim(self,x,y):
        return SequenceMatcher(None, x, y).ratio()

    def SimilarityCode2CommentB(self):
        return self.sim(self.old_code,self.old_comment)

    def SimilarityCode2CommentA(self):
        return self.sim(self.new_code,self.old_comment)

    def DistanceSimlaritesCode2Comment(self):
        pass

    def SimilarityChange2CommentB(self):
        pass

    def SimilarityChange2CommentA(self):
        pass

    def DistanceSimilarities(self):
        pass

    def extract(self):
        return [self.SimilarityCode2CommentB(),self.SimilarityCode2CommentA()]

'''
n_estimatorsint = 100, selected 100 Classification And Regression Trees (CART) as the basic classifiers 
max_features = 8, set the random subset value of the features as 8
criterion = 'gini', use 'Gini' function
'''

class DataSet:

    def __init__(self):
        self.old_comment_subtokens = None
        self.old_code_subtokens = None
        self.new_code_subtokens = None
        self.label = None

    def generate(self,dataset):
        data = self.get_raw_data(dataset)
        self.old_comment_subtokens = data[0]
        self.old_code_subtokens = data[1]
        self.new_code_subtokens = data[2]
        self.label = data[3]
    
    def get_raw_data(self,dataset):
        return ParamDataSet(dataset,
        "old_comment_subtokens", "old_code_subtokens","new_code_subtokens","label")


    def extractCommentFeature(self):
        result = list()
        for comment in self.old_comment_subtokens:
            cf = CommentFeatures(comment)
            result.append(cf.extract())
        return result

    def extractRelationshipFeature(self):
        result = list()
        for old_code,old_comment,new_code in zip(self.old_code_subtokens,self.old_comment_subtokens,self.new_code_subtokens):
            rf = RelationshipFeatures(old_code,new_code,old_comment)
            result.append(rf.extract())
        return result

    def data(self,dataset):
        x,y = list(),list()

        
        comment_feature = self.extractCommentFeature()

        relationship_feature = self.extractRelationshipFeature()

        x = np.array([x+y for x,y in zip(comment_feature,relationship_feature)])

        y = np.array(self.label)

        return x,y
    

if __name__ == "__main__":
    ds = DataSet()

    ds.generate('train')
    x_train,y_train = ds.data('train')

    ds.generate('test')
    x_test,y_test = ds.data('test')

    # model = RandomForestClassifier(n_estimators=100,criterion='gini',max_features=8)
    model = RandomForestClassifier(n_estimators=100,criterion='gini',max_features=3)

    # print(x_train)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))



