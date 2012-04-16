import os

sjm_deps = ['#', '#sift', '#util', '#codebooks', '#spatial_pyramid',
            '#naive_bayes_nearest_neighbor']

env = Environment(ENV = os.environ)

debug = ARGUMENTS.get('debug', 0)
profile = ARGUMENTS.get('profile', 0)
if not int(debug):
   env.Append(CCFLAGS = ['-O3', '-Wall'])
else:
   env.Append(CCFLAGS = ['-g', '-O0'])
if int(profile):
   env.Append(CCFLAGS = ['-pg'], LINKFLAGS=['-pg'])
env.Append(CCFLAGS = ['-std=c++0x', '-pedantic'])
env.Append(LIBPATH = [sjm_deps])
env.Append(CPPPATH = [sjm_deps])
env.Append(LIBS = ['glog', 'gflags'])

py_proto_builder = \
    Builder(action = 'protoc --python_out=./ $SOURCE',
            suffix = '_pb2.py',
            src_suffix = '.proto')

cc_proto_builder = \
    Builder(action = 'protoc --cpp_out=./ $SOURCE',
            suffix = '.pb.cc',
            src_suffix = '.proto')

h_proto_builder = \
    Builder(action = 'protoc --cpp_out=./ $SOURCE',
            suffix = '.pb.h',
            src_suffix = '.proto')

env.Append(BUILDERS = {'ProtoPy' : py_proto_builder})
env.Append(BUILDERS = {'ProtoCc' : cc_proto_builder})
env.Append(BUILDERS = {'ProtoH' : h_proto_builder})

test_env = env.Clone()
test_env.Append(LIBS = ['gtest', 'pthread'])

Export('env')
Export('test_env')

SConscript([
      'codebooks/SConscript',
      'sift/SConscript',
      'spatial_pyramid/SConscript',
      'naive_bayes_nearest_neighbor/SConscript',
      'naive_bayes_nearest_neighbor/experiment_1/SConscript',
      'naive_bayes_nearest_neighbor/experiment_3/SConscript',
      'util/SConscript',
      ])
