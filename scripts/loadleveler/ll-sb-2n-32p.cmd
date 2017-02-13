#@ wall_clock_limit = 01:00:00
#@ job_name = statinvasg_test 
#@ job_type = MPICH
#@ class = test
#@ output = statinvasg_test.out
#@ error = statinvasg_test.err
#@ node = 2
#@ total_tasks = 32
#@ node_usage = not_shared
#@ energy_policy_tag = impi_tests
#@ minimize_time_to_solution = yes
#@ island_count = 1
#@ queue

. /etc/profile
. /etc/profile.d/modules.sh

module load llvm
module load cmake
module load scons
module load python
module unload mpi.ibm
module load mpi.intel

export LD_LIBRARY_PATH=$HOME/workspace/statinvasg/lib/SGpp/lib/sgpp:$LD_LIBRARY_PATH

cd $HOME/workspace/statinvasg
echo $LD_LIBRARY_PATH
echo "Now start running..."
mpiexec -n 32 ./bin/run >> ./batch_ll/console.out
echo "Now done!"
