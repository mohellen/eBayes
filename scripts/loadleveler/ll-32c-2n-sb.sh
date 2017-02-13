#!/bin/bash
#@ wall_clock_limit = 00:05:00
#@ job_name = invasic-testing-32-simple-x10
#@ job_type = MPICH
#@ class = test
#@ output = 32-impi-tests_$(jobid).out
#@ error = 32-impi-tests_$(jobid).out
#@ node = 2
#@ total_tasks = 32
#@ node_usage = not_shared
#@ energy_policy_tag = impi_tests
#@ minimize_time_to_solution = yes
#@ island_count = 1
#@ queue

. /etc/profile
. /etc/profile.d/modules.sh
. $HOME/.bashrc

BATCH_USER=di29zaf2
# 16 cores per node, on the sandybridge (thin) nodes
# login with sb.supermuc.lrz.de
CORES_PER_NODE=16

# iMPI
export PATH=~/install/impi/bin:$PATH
export LD_LIBRARY_PATH=~/install/impi/lib:$LD_LIBRARY_PATH
which mpicc
# iRM
export PATH=~/install/irm/bin:$PATH
export PATH=~/install/irm/sbin:$PATH
export LD_LIBRARY_PATH=~/install/irm/lib:$LD_LIBRARY_PATH
# iRM
export PATH=~/install/munge/bin:$PATH
export PATH=~/install/munge/sbin:$PATH
export LD_LIBRARY_PATH=~/install/munge/lib:$LD_LIBRARY_PATH
export PATH=~/install/hwloc/bin:$PATH
export LD_LIBRARY_PATH=~/install/hwloc/lib:$LD_LIBRARY_PATH

# processing load-leveler host-file
cat $LOADL_HOSTFILE > host_file
echo "processing the Load Leveler provided hostfile "
echo "getting unique entries..."
awk '!a[$0]++' host_file > unique_hosts
cat unique_hosts > $LOADL_HOSTFILE
echo "new ll file:"
cat $LOADL_HOSTFILE 
echo "trimming the -ib part endings..."
rm -rf trimmed_unique_hosts
while read h; do
	echo $h | rev | cut -c 4- | rev >> trimmed_unique_hosts
done <unique_hosts

# generating slurm.conf dynamically
echo "copying initial slurm.conf work file"
cp -a ~/install/irm/etc/slurm.conf.in slurm.conf.initial
cp -a ~/install/irm/etc/slurm.conf.in slurm.conf.work
echo "setting up NodeName and PartitionName entries in slurm.conf ..."
while read h; do
	echo "NodeName=$h CPUs=${CORES_PER_NODE} State=UNKNOWN" >> slurm.conf.work
done <trimmed_unique_hosts
Nodes=`sed "N;s/\n/,/" trimmed_unique_hosts`
Nodes=`cat trimmed_unique_hosts | paste -sd "," -`
echo "PartitionName=local Nodes=${Nodes} Default=YES MaxTime=INFINITE State=UP" >> slurm.conf.work
FirstNode=`head -n 1 trimmed_unique_hosts`
echo "ControlMachine=${FirstNode}" >> slurm.conf.work
cp -a slurm.conf.work ~/install/irm/etc/slurm.conf

# starting the resource manager (slurm)
echo "starting daemons on each given node..."
n=`wc -l <unique_hosts`
autonomous_master.ksh -n $n -c "/home/hpc/h039w/${BATCH_USER}/install/munge/sbin/munged -Ff > munge_remote_daemon_out 2>&1" &
autonomous_master.ksh -n $n -c "/home/hpc/h039w/${BATCH_USER}/install/irm/sbin/slurmd -Dcvvvv > slurm_remote_daemon_out 2>&1" &
echo "starting the controller..."
irtsched -Dcvvvv > slurm_controller_out 2>&1 &
sleep 5

echo "exporting recommended variable..."
export SLURM_PMI_KVS_NO_DUP_KEYS
export MXM_LOG_LEVEL=error
export SLURM_MPI_TYPE=pmi2

echo "--------------------------------------------------------------------------------"
echo " Starnig the application with srun:"
date
srun -n 16 ./bin/adapt_with_wsloop > adapt_with_wsloop_output.txt
date
echo "--------------------------------------------------------------------------------"
