import pylearn2.scripts.train
import os

def run_stack(yaml_list):
  for yaml in yaml_list:
    yaml = os.getcwd() + "/" + yaml
    print "Training with %s" % yaml
    suffix_to_strip = '.yaml'
    config_file_name = ""
    if yaml.endswith(suffix_to_strip):
        config_file_name = yaml[0:-len(suffix_to_strip)]
    else:
        config_file_name = yaml
    # publish the PYLEARN2_TRAIN_FILE_NAME environment variable
    varname = "PYLEARN2_TRAIN_FILE_NAME"
    # this makes it available to other sections of code in this same script
    os.environ[varname] = yaml
    # this make it available to any subprocesses we launch
    os.putenv(varname, config_file_name)
    train_obj = pylearn2.config.yaml_parse.load(yaml)
    try:
      iter(train_obj)
      iterable = True
    except TypeError as e:
      iterable = False
      if iterable:
        for subobj in iter(train_obj):
          subobj.main_loop()
          del subobj
          gc.collect()
      else:
        train_obj.main_loop()
