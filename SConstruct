import os
import sys
import shutil
import SCons

###### Ininitialize env virables #######
########################################
cxx = 'mpicxx'
# Compile flags
cppflags = ['-O3','-std=c++11','-pedantic','-Wno-deprecated'] #'-fmessage-length=0','-Wno-unused-result'
# Include paths: -Iinclude without '-I'
cpppath = ['src', 'dep', 'dep/sgpp-base-2.0.0/base/src']
# Library look up paths: -Lpath without '-L'
libpath = []
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
vars.AddVariables(

    EnumVariable('scenario', 'Inverse problem scenarios: 1 for NS, 2 for heat', '1',
        allowed_values=('1') # so far only NaviarStokes supported
    ),
    EnumVariable('impi', 'Enable iMPI support [1|0]', '1',
        allowed_values=('1','0')
    ),
    PathVariable('impipath', 'Specify iMPI installation path', '/usr'),

    EnumVariable('impinodes', 'Enable iMPI node information in output [1|0]', '1',
        allowed_values=('1','0')
    ),
    EnumVariable('sgitime', 'Print time measurements during SGI surrogate construction [1|0]', '1',
        allowed_values=('1','0')
    ),
    EnumVariable('sgirank', 'Print rank progress during SGI surrogate construction [1|0]', '1',
        allowed_values=('1','0')
    ),
    EnumVariable('sgigps', 'Print grid point detail during SGI surrogate construction [1|0]', '0',
        allowed_values=('1','0')
    ),
    EnumVariable('mcmcprog', 'Print MCMC progress information [1|0]', '1',
        allowed_values=('1','0')
    ),
    EnumVariable('ealocal', 'Print rank local results for Error Analysis [1|0]', '0',
        allowed_values=('1','0')
    ),

    ( 'exe', 'Set executable name (default: main)', 'main' ),
)#end AddVariables
########################################


########## Setup environment ###########
########################################
env = Environment(variables=vars, ENV=os.environ)

env.Append( CCFLAGS=['-DIMPI='+env['impi']] )
env.Append( CCFLAGS=['-DIMPI_NODES='+env['impinodes']] ) #TODO: add node info support in src
env.Append( CCFLAGS=['-DGLOBAL_SCENARIO='+env['scenario']] )
env.Append( CCFLAGS=['-DSGI_PRINT_TIMER='+env['sgitime']] )
env.Append( CCFLAGS=['-DSGI_PRINT_RANKPROGRESS='+env['sgirank']] )
env.Append( CCFLAGS=['-DSGI_PRINT_GRIDPOINTS='+env['sgigps']] )
env.Append( CCFLAGS=['-DMCMC_PRINT_PROGRESS='+env['mcmcprog']] )
env.Append( CCFLAGS=['-DEA_LOCALINFO='+env['ealocal']] )

# check iMPI installation if it's enabled
if (env['impi']=='1'):
	if (os.path.isfile(env['impipath'] + '/bin/mpicc') and
			os.path.isfile(env['impipath'] + '/lib/libmpi.so')):
            impipath = env['impipath']
        elif (os.path.isfile(os.environ['IMPIPATH'] + '/bin/mpicc') and 
                os.path.isfile(os.environ['IMPIPATH'] + '/lib/libmpi.so')):
            impipath = os.environ['IMPIPATH']
        else:
            sys.exit("Error: iMPI installation not found. Check impipath. Operation aborted.")
        cpppath += [impipath + '/include']
        libpath += [impipath + '/lib']
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
builddir = bindir + '/build_' + tar
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


############### PHONY TARGETS ################
##############################################
def PhonyTargets(env = None, **kw):
    if not env: 
		env = Environment(variables=vars, ENV=os.environ)
    for target,action in kw.items():
        env.AlwaysBuild(env.Alias(target, [], action))

## [Usage]
## Example 1: simple target, default env
## PhonyTargets(TAGS = 'tools/mktags.sh -e')
##
## Example 2: multple targets
## env = Environment(parse_flags = '-std=c89 -DFOO -lm')
## PhonyTargets(env, CFLAGS  = '@echo $CFLAGS',
##                   DEFINES = '@echo $CPPDEFINES',
##                   LIBS    = '@echo $LIBS')

cmd = 'cd '+basedir+'/dep/sgpp-base-2.0.0; scons -c; scons BUILDDIR='+impipath+'/lib; cd '+basedir
PhonyTargets(sgpp = cmd)
##############################################

