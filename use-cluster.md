### Connect to the cluster

- sudo gpclient connect gp-dmat-saml.vpn.polimi.it
- ping 10.78.18.100 (to check connection)
- ssh u11177200@10.78.18.100

### Check the cluster status and request an interactive session

- qstat -Q
- pbsnodes -aSj
- qsub -I -q cpu -l select=1:ncpus=6:mpiprocs=2:host=cpu05

  - -q cpu : queue name
  - than we have the number of nodes (1), number of cpus (6) and number of mpi processes (2) (threads). and the specified cpu

apptainer shell ../amsc_mk_2025.sif
source /u/sw/etc/bash.bashrc
module load gcc-glibc dealii

### tmux session

- tmux new -s my_session
- inside we do everything :
  - qsub -I -q cpu -l select=1:ncpus=16 walltime=24:00:00
  - apptainer shell ../amsc_mk_2025.sif
  - source /u/sw/etc/bash.bashrc
  - module load gcc-glibc dealii
  - run
- to detach from the session : Ctrl + b , d (keep it alive for later)
- to reattach : tmux attach -t my_session
