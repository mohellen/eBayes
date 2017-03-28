import os
import shutil
import SCons

###### Ininitialize env virables #######
########################################
# Include paths: -Iinclude without '-I'
cxx = 'mpicxx'
cpppath = ['src', 'include', 'dep/sgpp-base-2.0.0/base/src']
# Compile flags
cppflags = ['-O3','-g','-std=c++11','-fmessage-length=0',
            '-Wno-unused-result','-Wno-deprecated','-pedantic']
# Library look up paths: -Lpath without '-L'
libpath = ['lib']
# Libraries to link: -lmylib without '-l'
libs = ['m','sgppbase','mpi','mpicxx']
########################################


#### Customs command line variables ####
########################################
vars = Variables()
vars.Add(BoolVariable("ENABLE_IMPI", "Enable elastic MPI (default: 0)", False))
vars.Add("IMPI_ADAPT", "Set iMPI resource adaptation frequency (defalt: 30)", 30)
vars.Add("EXEC", "Set executable name (default: main)", "main")
########################################


########## Setup environment ###########
########################################
env = Environment(variables=vars, ENV=os.environ)
env.Replace(CXX=cxx)
env.Append(CPPPATH=cpppath)
env.Append(CPPFLAGS=cppflags)
env["ENABLE_IMPI"] = env.get("ENABLE_IMPI")
env["IMPI_ADAPT"] = env.get("IMPI_ADAPT")

if (env["ENABLE_IMPI"]):
    libpath += ['/media/data/nfs/install/lib']
    env.Append(CPPPATH='/media/data/nfs/install/include')
    env.Append( CPPDEFINES=['ENABLE_IMPI=1'] )
else:
    libpath += ['/media/data/install/mpich-3.2/lib']
    env.Append(CPPPATH='/media/data/install/mpich-3.2/include')
    env.Append( CPPDEFINES=['ENABLE_IMPI=0'] )
    
env.Append( CPPDEFINES=['IMPI_ADAPT=' + str(env["IMPI_ADAPT"])]   )
########################################


############ Help Message ##############
########################################
env.Help("""
Usage: 
To build with defalt options: scons
To build with special options: scons <OPTION>=<VALUE>
To clean up build: scons -c

Available options are:
"""+vars.GenerateHelpText(env)
)
########################################


################ BUILD #################
########################################
# Specify build name
BuildName = env.get("EXEC")
# Save the base path
BASEPATH = os.getcwd()
BUILDPATH = BASEPATH + '/bin/' + BuildName + '_build'
TARGET = BASEPATH + '/bin/' + BuildName
# Make output and build dir
if not os.path.exists(BASEPATH + '/output'):
    os.makedirs(BASEPATH + '/output')
if not os.path.exists(BUILDPATH):
    os.makedirs(BUILDPATH)
# List of objects for the build
OBJ = []
# Build objects from sources file
# find each subdir in src            
for sdir in os.listdir('src'):
    srcdir = BASEPATH + '/src/' + sdir
    # confirm this is a dir (not a file), then
    if (os.path.isdir(srcdir)):
        # create the corresponding build path
        blddir = BUILDPATH + '/' + sdir
        if not os.path.exists(blddir):
            os.makedirs(blddir)
        # find all source files in this dir
        srcfiles = []
        for f in os.listdir('src/'+sdir):
            if (f.endswith('.cpp') or f.endswith('.c') or f.endswith('.cc')):
                # NOTE: Must prefix "build dir" to each source file in order 
                #       to be built in the build dir
                srcfiles += [blddir +'/'+ f]
                # Put the object file into object list
                OBJ += [ os.path.splitext(blddir +'/'+ f)[0]+'.o' ]
        # compile all files
        env.VariantDir(blddir, srcdir, duplicate=0)
        env.Object(srcfiles)
    # if sdir is indeed a file, check if it's a source file
    elif (sdir.endswith('.cpp') or sdir.endswith('.c') or sdir.endswith('.cc')):
        srcdir = BASEPATH + '/src'
        blddir = BUILDPATH
        srcfile = blddir +'/'+ sdir
        OBJ += [ os.path.splitext(blddir +'/'+ sdir)[0]+'.o' ]
        # compile the file
        env.VariantDir(blddir, srcdir, duplicate=0)
        env.Object(srcfile)
# Build program
env.Program(TARGET, OBJ, LIBS=libs, LIBPATH=libpath)
########################################


############### CLEANUP ################
########################################
env.Clean("clean", [TARGET, BUILDPATH])
########################################

# TODO: make phony target 'scons sgpp' 

#sgpp:
#    mkdir -p $(BASEPATH)/lib; cd $(BASEPATH)/dep/sgpp-base-2.0.0; scons -c; scons BUILDDIR=$(BASEPATH)/lib -j4; cd $(BASEPATH)







