from jobman import api0, sql, expand
from jobman.tools import DD, flatten

# Experiment function
import my_script

# Database
TABLE_NAME = 'JULIENS_023_DAE'
db = sql.db('postgres://ift6266h12@gershwin.iro.umontreal.ca/ift6266h12_db/'+TABLE_NAME)

# Default values
state = DD()
state['jobman.experiment'] = 'my_script.execute'

# Parameters
learning_rate_values = [0.005, 0.01]
nb_hidden_values = [575, 600, 625]
corrupt_levels = [0.1, 0.4, 0.5]
n_comp_trans_pca = [10, 11, 12, 15, 17]

state['tied_weights'] = True
state['max_epochs'] = 50
state['work_dir'] = '/data/lisa/exp/ift6266h12/juliensi/repos/JulienS-ift6266h12/ift6266h12/results/avicenna/'


for learning_rate in learning_rate_values:
    for nhid in nb_hidden_values:
        for corruption_level in corrupt_levels:
            for n_components_trans_pca in n_comp_trans_pca:
        
                state['n_components_trans_pca'] = n_components_trans_pca
                state['corruption_level'] = corruption_level
                state['learning_rate'] = learning_rate
                state['nhid'] = nhid

                sql.add_experiments_to_db([state],db, verbose=1,force_dup=True)
