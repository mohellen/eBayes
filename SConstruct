import os
import shutil
import SCons

###### Ininitialize env virables #######
########################################
cxx = 'mpicxx'
# Compile flags
cppflags = ['-O3','-std=c++11','-fmessage-length=0',
            '-Wno-unused-result','-Wno-deprecated','-pedantic']
# Include paths: -Iinclude without '-I'
cpppath = ['src', 'include', 'dep/sgpp-base-2.0.0/base/src']
# Library look up paths: -Lpath without '-L'
libpath = ['lib']
# Libraries to link: -lmylib without '-l'
libs = ['m','sgppbase','mpi','mpicxx']
########################################


#### Local variables ####
#########################
homedir = os.path.expanduser("~")
basedir = os.getcwd()
srcdir = basedir + "/src"
bindir = basedir + "/bin"
#########################


#### Customs command line variables ####
########################################
vars = Variables()
vars.Add( BoolVariable('impi', 'Enable iMPI support', True) )
vars.Add( PathVariable('impipath', 'Specify iMPI installation path', homedir + '/workspace/ihpcins') )
vars.Add( BoolVariable('impinodes', 'Enable iMPI node information in output', True) )
vars.Add( 'exe', 'Set executable name (default: main)', 'main' )
########################################


########## Setup environment ###########
########################################
env = Environment(variables=vars, ENV=os.environ)

if env['impi']:
	env.Append( CCFLAGS=['-DIMPI'] )
	cpppath += [env['impipath'] + '/include']
	libpath += [env['impipath'] + '/lib']
	if env['impinodes']:
		env.Append( CCFLAGS=['-DIMPI_NODES'] )
else:
	cpppath += ['/usr/include/mpich']
	libpath += ['/usr/lib']

env.Replace( CXX=cxx )
env.Append( CPPPATH=cpppath )
env.Append( LD_LIBRARY_PATH=libpath )
env.Append( CPPFLAGS=cppflags )
########################################


############ Help Message ##############
########################################
env.Help("""
[Usage]

To build with defalt options:
	scons

To build with special options:
	scons <OPTION>=<VALUE>

To build with options specified by config file:
	scons config=/path/to/configfile.py

To clean up build:
	scons -c

[Available options]
"""+vars.GenerateHelpText(env)
)
########################################


################ BUILD #################
########################################
# Specify build name
tar = str(env['exe'])
# Save the base path
target = bindir + '/' + tar
builddir = bindir + '/' + tar + '_build'
# Make output and build dir
if not os.path.exists(bindir):
	os.makedirs(bindir)
if not os.path.exists(builddir):
	os.makedirs(builddir)

# Find (recursively) all source paths
def recursive_add_paths(thepath, pathlist):
	for p in os.listdir(thepath):
		# For some reason, os.path.abspath(p) return wrong results
		absp = thepath + '/' + p
		if (os.path.isdir(absp)):
			pathlist += [absp]
			recursive_add_paths(absp, pathlist)

srcpaths = [srcdir]
recursive_add_paths(srcdir, srcpaths)

# Build objects from source files
objs = []
for p in srcpaths:
	# create corresponding build path
	destp = p.replace(srcdir, builddir)
	if not os.path.exists(destp):
		os.makedirs(destp)
	# for each source file in p
	for f in os.listdir(p):
		absf = p + '/' + f
		if (os.path.isfile(absf) and (absf.endswith('.cpp') or absf.endswith('.cc') or absf.endswith('.c'))):
			# define corresponding target object
			tarf = destp + '/' + f
			tarf = [os.path.splitext(tarf)[0]+'.o']
			objs += [tarf]
			# compile the file
			env.Object(tarf, absf)

# Build program
env.Program(target, objs, LIBS=libs, LIBPATH=libpath)
########################################



############### CLEANUP ################
########################################
env.Clean("clean", [target, builddir])
########################################

# TODO: make phony target 'scons sgpp' 

#sgpp:
#    mkdir -p $(BASEPATH)/lib; cd $(BASEPATH)/dep/sgpp-base-2.0.0; scons -c; scons BUILDDIR=$(BASEPATH)/lib -j4; cd $(BASEPATH)







