import pylearn2.scripts.train
import os
from pylearn2.utils import serial
'''
Run the yaml files in the list one by one from the path.
'''
def run_stack(yaml_list):
  for yaml in yaml_list:
    run_on_layer_file(yaml)

'''
Run the training on yaml file
'''
def run_on_layer_file(config_file_path):
  print "Training with %s" % config_file_path
  suffix_to_strip = '.yaml'
  yaml_file = open(config_file_path, 'r')
  if config_file_path.endswith(suffix_to_strip):
      config_file_name = config_file_path[0:-len(suffix_to_strip)]
  else:
      config_file_name = config_file_path

  # publish the PYLEARN2_TRAIN_FILE_NAME environment variable
  varname = "PYLEARN2_TRAIN_FILE_NAME"
  # this makes it available to other sections of code in this same script
  os.environ[varname] = config_file_name
  print config_file_name
  # this make it available to any subprocesses we launch
  os.putenv(varname, config_file_name)
  train_obj = pylearn2.config.yaml_parse.load(yaml_file)

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

def run_on_layer_string(yaml_str, varname):
  suffix_to_strip = '.yaml'
  yaml_file = open(config_file_path, 'r')
  if config_file_path.endswith(suffix_to_strip):
      config_file_name = config_file_path[0:-len(suffix_to_strip)]
  else:
      config_file_name = config_file_path

  # publish the PYLEARN2_TRAIN_FILE_NAME environment variable
  varname = "PYLEARN2_TRAIN_FILE_NAME"
  # this makes it available to other sections of code in this same script
  os.environ[varname] = config_file_name
  print config_file_name
  # this make it available to any subprocesses we launch
  os.putenv(varname, config_file_name)
  train_obj = pylearn2.config.yaml_parse.load(yaml_file)

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

#'''Test
config = 'yamls/cae1.yaml'
run_on_layer(config)
#'''
