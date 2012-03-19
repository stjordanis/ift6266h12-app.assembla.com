from ift6266h12.dae_avicenna import main_train

def execute(state, channel):
    work_dir = state['work_dir']
    corruption_level = state['corruption_level']
    nhid = state['nhid']
    tied_weights = state['tied_weights']
    max_epochs = state['max_epochs']
    learning_rate = state['learning_rate']
    n_components_trans_pca = state['n_components_trans_pca']
    

    out = main_train(work_dir = work_dir, corruption_level = corruption_level, nhid = nhid, tied_weights = tied_weights, max_epochs = max_epochs, learning_rate = learning_rate, n_components_trans_pca = n_components_trans_pca)

    state['timespent'] = out[0]
    state['alc1'] = out[1]
    state['alc2'] = out[2]
    
    channel.save()

    return channel.COMPLETE