from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    ## please add the above item to the counterpart file in your pytracking toolkit
    settings.itb_path = 'path to ITB' # for example /home/data/testing_dataset/ITB'


    return settings

