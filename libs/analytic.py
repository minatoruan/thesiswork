from multiprocessing import Pool
from sklearn.metrics import * 

# Code for Comparations of Classifiers
def estimate(args):
    estimator, train_set, test_set = args[0], args[1], args[2]
    estimator.fit(train_set)
    return [[_id, pred] for _id, pred in zip(test_set.get('ID'), estimator.predict(test_set))]

def cv_estimate(estimator, splitter, n_jobs = 4):
    y_all_ids = splitter.kfold_data().get('ID sw_annotation')
    ids = (y_all_ids[:,0].reshape(-1)).tolist()
    y_all_test = (y_all_ids[:,1].reshape(-1)).tolist()
    y_all_pred = (y_all_ids[:,1].reshape(-1)).tolist()
    pool = Pool(n_jobs)
    results = pool.map(estimate, [(estimator, train_set, test_set) for test_set, train_set in splitter.get()])
    
    for re in results:
        for item in re:
            y_all_pred[ids.index(item[0])] = item[1]
    print 'Cross validation score', 
    print 'Confusion matrix\n', confusion_matrix(y_all_test, y_all_pred)
    print 'Accuracy score\n', accuracy_score(y_all_test, y_all_pred)
    pool.close()
    
    estimator.fit(splitter.kfold_data())
    y_t = splitter.get_testing().get('sw_annotation')
    y_p = estimator.predict(splitter.get_testing())
    
    print 'Testing score'
    print 'Confusion matrix\n', confusion_matrix(y_t, y_p)
    print 'Accuracy score\n', accuracy_score(y_t, y_p)
    
    return y_t, y_p

# Code for Nofaultvsfault_threshold_study
def estimate1(args):
    estimator, train_set, test_set = args[0], args[1], args[2]
    estimator.fit(train_set)
    return [[_id, pred] for _id, pred in zip(test_set.get('ID'), estimator.predict(test_set.todict()[0]))]

def cv_estimate_kfold(estimator, splitter, n_jobs = 4):
    y_all_ids = splitter.getinterpretor().get('ID nofaultvsfault')
    ids = (y_all_ids[:,0].reshape(-1)).tolist()
    y_all_test = (y_all_ids[:,1].reshape(-1)).tolist()
    y_all_pred = (y_all_ids[:,1].reshape(-1)).tolist()
    pool = Pool(n_jobs)
    results = pool.map(estimate1, [(estimator, train_set, test_set) for test_set, train_set in splitter.get()])
    
    for re in results:
        for item in re:
            y_all_pred[ids.index(item[0])] = item[1]
    print 'Cross validation score', 
    print 'Confusion matrix\n', confusion_matrix(y_all_test, y_all_pred)
    print 'Accuracy score\n', accuracy_score(y_all_test, y_all_pred)
    pool.close()
    
    return y_all_test, y_all_pred

#http://weka.sourceforge.net/doc.dev/weka/classifiers/trees/J48.html
# holding a setting for training a classifer
class WekaJ48Result:
    @staticmethod                           
    def writeCSVHeaderTo(writer):
        template = 'minNumObj;properties;leaves;ground_accuracy;ground_false_neg;ground_false_pos;line_accuracy;line_false_neg;line_false_pos;phase_accuracy;phase_false_neg;phase_false_pos;nofault_accuracy;nofault_false_neg;nofault_false_pos;accuracy'
        writer.writerow(template.split(';'))
    
    @staticmethod    
    def parse(iterobj):
        a_cfm = []
        while (True):
            try:
                line = iterobj.next()
                if (line.startswith('Options:')): 
                    strips = line.split(':')
                    j45properties = strips[1]
                    print 'Options:', j45properties
                    continue
                if (line.startswith('Number of Leaves  :')): 
                    strips = line.split(':')
                    nLeaves = int(strips[1].strip())
                    print 'Number of leaves:', nLeaves
                    continue
                if (line == '=== Confusion Matrix ==='):
                    iterobj.next()
                    iterobj.next()
                    
                    arr = [[0 for x in xrange(4)] for x in xrange(4)]
                    for i in xrange(4):
                        line = iterobj.next()
                        for j, v in enumerate([int(x) for x in line.split(' ') if x.isdigit()]):
                            arr[i][j] = v
                    a_cfm.append(arr)
                    continue
            except StopIteration:
                break       
        return WekaJ48Result(j45properties, nLeaves, a_cfm[0], a_cfm[1])
    
    @property
    def cfm(self): 
        return self._cfm
    
    @property
    def cfmAccuracy(self): 
        return self._cfmAccuracy
    
    @property
    def leaves(self): 
        return self._leaves
    
    def __init__(self, properties, leaves, accuracy, crossValidateArr):
        self._properties = properties
        self._leaves = leaves
        self._cfm = cfm(crossValidateArr)
        self._cfmAccuracy = cfm(accuracy)