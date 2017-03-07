import os
import SCons

### Ininitialize env virables
########################################
# Include paths: -Iinclude without '-I'
cpppath = ['src', 'include', 'dep/sgpp-base-2.0.0/base/src']
# Compile flags
cppflags = ['-O3','-g','-std=c++11','-length=0',
            '-Wno-unused-result','-Wno-deprecated','-pedantic','-fopenmp']
# Library look up paths: -Lpath without '-L'
libpath = ['lib']
# Libraries to link: -lmylib without '-l'
libs = ['m','sgppbase','mpi','mpicxx']
########################################


### Customs command line variables
########################################
vars = Variables()
vars.Add(BoolVariable("ENABLE_IMPI", "Enable elastic MPI (default: 0)", False))
vars.Add("IMPI_ADAPT", "Set source adaptation frequency (defalt: 30)", 30)
vars.Add("OUTPUT_PATH", "Set path of output files (default: output)", "\"./out\"")
########################################


### Setup environment
########################################
env = Environment(variables=vars, ENV=os.environ)
env.Append(CPPPATH=cpppath)
env.Append(CPPFLAGS=cppflags)
env["ENABLE_IMPI"] = env.get("ENABLE_IMPI")
env["IMPI_ADAPT"] = env.get("IMPI_ADAPT")
env["OUTPUT_PATH"] = env.get("OUTPUT_PATH")

if (env["ENABLE_IMPI"]):
    env.Append(CPPPATH='/media/data/nfs/install/include')
    libpath += ['/media/data/nfs/install/lib']
else:
    env.Append(CPPPATH='/media/data/install/mpich-3.2/include')
    libpath += ['/media/data/install/mpich-3.2/lib']
    
env.Append( CPPDEFINES=['ENABLE_IMPI=' + str(env["ENABLE_IMPI"])] )
env.Append( CPPDEFINES=['IMPI_ADAPT=' + str(env["IMPI_ADAPT"])]   )
#env.Append( CPPDEFINES=['OUTPUT_PATH=' + str(env["OUTPUT_PATH"])] )
########################################


################ BUILD #################
########################################
# Specify build name
BuildName = 'main'
# Save the base path
BASEPATH = os.getcwd()
BASEBLDPATH = BASEPATH + '/bin/' + BuildName + '_build'
# Make bin and bin/<build> if not exists
if not os.path.exists(BASEPATH + '/bin'):
    os.makedirs(BASEPATH + '/bin')
if not os.path.exists(BASEBLDPATH):
    os.makedirs(BASEBLDPATH)
# List of objects for the build
OBJ = []
# Build objects from sources file
# find each subdir in src            
for sdir in os.listdir('src'):
    srcpath = BASEPATH + '/src/' + sdir
    # confirm this is a dir (not a file), then
    if (os.path.isdir(srcpath)):
        # create the corresponding build path
        bldpath = BASEBLDPATH + '/' + sdir
        if not os.path.exists(bldpath):
            os.makedirs(bldpath)
        # find all source files in this dir
        srcfiles = []
        for f in os.listdir('src/'+sdir):
            if (f.endswith('.cpp') or f.endswith('.c') or f.endswith('.cc')):
                # NOTE: Must prefix "build dir" to each source file in order 
                #       to be built in the build dir
                srcfiles += [bldpath +'/'+ f]
                # Put the object file into object list
                OBJ += [ os.path.splitext(bldpath +'/'+ f)[0]+'.o' ]
        # compile all files
        env.VariantDir(bldpath, srcpath, duplicate=0)
        env.Object(srcfiles)
    # if sdir is indeed a file, check if it's a source file
    elif (sdir.endswith('.cpp') or sdir.endswith('.c') or sdir.endswith('.cc')):
        srcpath = BASEPATH + '/src'
        bldpath = BASEBLDPATH
        srcfile = bldpath +'/'+ sdir
        OBJ += [ os.path.splitext(bldpath +'/'+ sdir)[0]+'.o' ]
        # compile the file
        env.VariantDir(bldpath, srcpath, duplicate=0)
        env.Object(srcfile)
# Build program
env.Program(BASEPATH+'/bin/'+BuildName, OBJ,
            LIBS=libs, LIBPATH=libpath)
########################################

