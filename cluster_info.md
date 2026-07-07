Technical Summary

The Cosmos supercomputer is built on the HPE Cray Supercomputing EX2500 platform, incorporating innovative AMD Instinct™ MI300A accelerated processing units (APUs), HPE Slingshot interconnect and a flash-based filesystem.  The APU uniquely features an in-chip memory layout, which is integrated and shared between CPU and GPU resources. This type of memory architecture facilitates an incremental programming approach, which enables many communities to adopt GPUs and ease the process of porting and optimizing a range of applications. The high-performance VAST filesystem incorporates flash-based storage and provides the high IOPS, and bandwidth needed for the anticipated mixed-application workload

Cosmos is an NSF-funded system, developed in collaboration with CRAY and operated by the San Diego Supercomputer Center at UC San Diego.

Resource Allocation Policies
Current Status: Testbed Phase
3-year testbed phase will be available to select focused projects, as well as workshops and industry interactions.
The testbed phase will be followed up with a 2 year allocation phase to the broader NSF community and User workshops.
To get access to Cosmos, please send a request to HPC Consulting.
All user must review and agree to Cosmos AUP. 
Technical Details
42 nodes, each with 4 APUs in a fully connected network based on AMDs Infinity xGMI (socket-to-socket global memory interface) technology, which provides 768 GBps aggregate and 256 GBps peer-to-peer bi-directional bandwidth between APUs. xGMI is the equivalent of NVIDIA’s NVLink. The cluster is managed by the SLURM scheduler, which orchestrates job distribution and execution.
168 AMD MI300A APUs. The MI300A combinesx86 CPU cores, CDNA3 GPU compute +shared memory access between CPU and GPU. Each APU has a theoretical peak performance of 90 fp64 (HPC) TFLOPS and 760 fp16 (AI) PFLOPS. Accounting for power limits, the whole system will provide close to 10 HPC PFLOPS and 100 AI PFLOPS of usable performance.
A high-performance interconnect based on HPE’s Slingshot technology, which provides low latency and congestion control.
300 TB of high-performance storage from VAST that provides the high IOPS and bandwidth needed for the anticipated mixed-application workload.
5 PB of Ceph capacity storage to provide excellent I/O performance for most applications and to store persistent project data.
Home File System via access to SDSC’s Qumulo storage, which provides a highly reliable, snapshotted file system.
Cosmos Architecure Overview

Technical Summary
System ComponentConfiguration
HPE EX2500 MI300A Compute Nodes
MI300A APUs168
APUs/node4
Nodes/blade2
EPYC Zen4 cores per APU24
CDNA4 GPU cores per APU228
Interconnect
TopologyDragonfly
NetworkSlingshot (ethernet-based)
Link Bandwidth(bi-directional)200GB/s
Performance Storage
Capacity200TB (usable)
Bandwidth (R-W)100:10 GB/s
IOPS676,000
File systemVAST NSF
NetworkEthernet
Capacity storage
Capacity5 PB (usable)
Bandwidth25 GB/s
File systemsCeph
NetworkEthernet
Home File System Storage
Capacity200 TB (expandable)
FilesystemQumulo NFS
NetworkEthernet
Service Nodes and Switching Rack
Login Nodes2
Admin and Fabric Nodes4
Top of rack Slighshot Switch1
Top of rack Aruba management and space switches4
1Gbe management switch1
Systems Software Environment
Systems Software Environment
Software FunctionDescription
Cluster ManagementHPE Performance Cluster Manager (HPCM)
Operating SystemSUSE Linux Enterprise Server
File SystemsCeph, Qumulo, VAST
Scheduler and Resource Manager SLURM
User EnvironmentHPC Cray programming environment
System Access

Logging in to Cosmos
Cosmos uses ssh key pairs for access. Approved users will need to send their ssh public key to consult@sdsc.edu to gain access to the system.  To log in to Cosmos from the command line, use one of the following hostnames:

login01.cosmos.sdsc.edu
login02.cosmos.sdsc.edu
cosmos01.cosmos.sdsc.edu
cosmos02.cosmos.sdsc.edu
The following are examples of Secure Shell (ssh) commands that may be used to log in to Cosmos:

ssh <your_username>@login01.cosmos.sdsc.edu
ssh -l <your_username>login01.cosmos.sdsc.edu

NOTES AND HINTS
Cosmos will not maintain local passwords,  your public key will need to be appended to your ~/.ssh/authorized_keys file to enable access from authorized hosts. We accept RSA, ECDSA and ed25519 keys. Make sure you have a strong passphrase on the private key on your local machine.
You can use ssh-agent or keychain to avoid repeatedly typing the private key password.
Hosts which connect to SSH more frequently than ten times per minute may get blocked for a short period of time
For Windows Users, you can follow the exact same instructions using either PowerShell, Windows Subsystems for Linux (WSL) (a compatibility layer introduced by Microsoft that allows users to run a Linux environment natively on a Windows system without the need for a virtual machine or dual-boot setup), or terminal emulators such as Putty or MobaXterm.

Do not use the login node for computationally intensive processes, as hosts for running workflow management tools, as primary data transfer nodes for large or numerous data transfers or as servers providing other services accessible to the Internet. The login nodes are meant for file editing, simple data analysis, and other tasks that use minimal compute resources. All computationally demanding jobs should be run using kubernetes.
Adding Users to a Project
Approved Cosmos project PIs and co-PIs can add/remove users(accounts) to/from a Cosmos. Please submit a support ticket to consult@sdsc.edu to add/remove users.

Modules

Cosmos uses the Environment Modules package to control user environment settings. Below is a brief discussion of its common usage. The Environment Modules package provides for dynamic modification of a shell environment. Module commands set, change, or delete environment variables, typically in support of a particular application. They also let the user choose between different versions of the same software or different combinations of related codes.

To check for currently available versions please use the command:

 avail
Useful Modules Commands
Here are some common module commands and their descriptions:

CommandDescription
module list
List the modules that are currently loaded

module avail
List the modules that are available in environment

module display <module_name>
Show the environment variables used by <module name> and how they are affected

module unload <module name>
Remove <module name> from the environment

module load <module name>
Load <module name> into the environment

module swap <module one> <module two>
Replace <module one> with <module two> in the environment


Running Jobs on Cosmos

Cosmos uses the Simple Linux Utility for Resource Management (SLURM) batch environment. When you run in the batch mode, you submit jobs to be run on the compute nodes using the sbatch command as described below. At present a single compute node with 4 APUs or multiple compute nodes can be requested. Compute nodes are not shared.  Remember that computationally intensive jobs should be run only on the compute nodes and not the login nodes.

The MI300A APU can be partitioned in 3 different modes:

1) SPX (Single Partition): All 6 XCDs are grouped into one partition and we get 4 “GPUs” per node

2) CPX (Core Partitioned): Each XCD is treated as a separate partition, yielding 6 partitions per socket and 24 “GPUs” per node, and

3) TPX (Triple Partition): three partitions, each containing two XCDs and providing three logical GPU devices per socket 12 “GPUs” per node.

Currently, the default setting on Cosmos is the SPX mode and a standard slurm submission will give you this option. There are 6 nodes reserved in the “TPX” mode that can be accessed by adding the following reservation line:

#SBATCH --res=tpx


Requesting interactive resources using srun
You can request an interactive session using the srun command. The following example will request one regular compute node with 4 MI300A APUs for 30 minutes.

srun --pty --nodes=1 -t 00:30:00 --exclusive /bin/bash
In this example:

--pty: Allocates a pseudo-terminal.
--nodes=1: Requests one node.
-t 00:30:00: Sets the time limit to 30 minutes.
--cpus-per-task=96: 96 cores per task (e.g. can use for threads)
/bin/bash: Opens a Bash shell upon successful allocation.
 
Submitting Jobs Using sbatch
Jobs can be submitted to the sbatch partitions using the sbatch command as follows:

 sbatch jobscriptfile
where jobscriptfile is the name of a UNIX format file containing special statements (corresponding to sbatch options), resource specifications and shell commands. Several example SLURM scripts are given below:

BASIC JOB
#!/bin/bash
#SBATCH --job-name=hellompi
#SBATCH --output=hellompi.%j.%N.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH -t 04:00:00

./hello_world
Basic Job using Containers and AMD ROCm compiler
#!/bin/bash
#SBATCH --job-name=hello_openmp
#SBATCH --output=hello_openmp.%j.%N.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH -t 01:30:00

### Example from https://github.com/ROCm/MAD
### Token info not included
### Load Singulariy module 
module load singularitypro

### Run vllm script
### --rocm to get the drivers
### --bind to mount external filesystems into container
singularity exec --rocm --bind /scratch/nchen9:/workspace /cosmos/nfs/home/nchen9/examples/hf/vllm_rocm.sif ./vllm_benchmark_report.sh -s all -m meta-llama-3.1-*B-Instruct -g 4 -d float 8

### Clean up /scratch
rm -r /scratch/nchen9/modules*

BASIC MPI JOB
#!/bin/bash
#SBATCH --job-name=hellohybrid
#SBATCH --output=hellohybrid.%j.%N.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH -t 01:30:00

scontrol show hostname
cd osu-micro-benchmarks-7.5-1/c/mpi/pt2pt/standard
srun -n 2 --ntasks-per-node 1 ./osu_latency
srun -n 2 --ntasks-per-node 1 ./osu_bw
Job Monitoring and Management
Users can monitor jobs using the squeue command.

[user ~]$ squeue -u user1

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            256556   cluster vllmslu  user1     R    2:03:57      1 x8000c2s1b0n0
 

Users can cancel their own jobs using the scancel command as follows:

[user ~]$ scancel <jobid> 
Information on the Partitions
[user ~]$ sinfo

Commonly used commands in Slurm:

Below you can see a table with a handful useful commands that are often used to check the status of submitted jobs:

 

Commonly used commands in Slurm
Action

Slurm command

Interactive Job submission

srun

Batch Job submission

sbatch jobscript

List user jobs and their nodes

squeue –u nchen9

Job deletion

scancel <job-id>

Job status check

scontrol show job <job-id>

Node availability check

sinfo

Usage Guidelines
There are currently no enforced resource limits on Cosmos during the Early User Period. Formal limits will be introduced when Cosmos transitions to its production phase, based on evaluation and feedback from early users.

Although no hard limits are in place, users are expected to follow the partition guidelines listed in the table below. Because Cosmos has a limited number of nodes, keeping job sizes modest in terms of node count and walltime helps ensure fair and efficient scheduling for all users.

During standard business hours, please leave sufficient capacity available for interactive debugging and testing so that all users can effectively support development workflows.

If you plan to run long-duration or large multi-node jobs, please contact consult@sdsc.edu in advance. This helps us coordinate scheduling and minimize impact on other users.

Suggested Usage Guidelines
Resource NameMax
WalltimeMax
Nodes/JobNotes
cluster_info.md 24 hrs4Suggested limits for Early User testing.
Data Movement

Globus Endpoints, Data Movers and Mount Points  (** Coming Soon)
 

Storage

Overview
Users are responsible for backing up all important data to protect against data loss at SDSC.

Home File System:
The user home directories on Cosmos utilize a NFS mounted Qumulo filesystem. However, it has some limitations, and proper usage is essential for optimized performance and to prevent filesystem overloads. Details of the home filesystem are as follows:

LOCATION AND ENVIRONMENT VARIABLE
After logging in, you'll find yourself in you home directory.
This directory is also accessible via the environment variable /cosmos/nfs/home/nchen9.
STORAGE LIMITATIONS AND QUOTA
The home directory comes with a storage quota of 100GB.
It is not meant for large data storage or high I/O operations.
WHAT TO STORE IN THE HOME DIRECTORY
You should only use the home directory for source code, binaries, and small input files.
WHAT NOT TO DO
Avoid running jobs that perform intensive I/O operations in the home directory.
For jobs requiring high I/O throughput, it's better to use VAST filesystem or the node local scratch space.
VAST File System
The performance storage component of Cosmos is an all-flash file system from VAST. All users have a VAST storage location at /cosmos/vast/scratch/nchen9. The VAST filesystem provides high IOPS and bandwidth and will be useful for metadata intensive workloads and for scalable IO workloads. The filesystem is limited in size and primarily intended to be used for scratch storage needs of running jobs or for use in active workflows combining many jobs. Any long-term storage needs must be addressed by using the Ceph (Coming Soon) system.

 

Node Local NVMe-based Scratch File System
Each Cosmos node has a 1.9-TB NVMe drive for use by jobs that benefit from a local scratch file system.  Job specific directory is created at the start of a job at /scratch/nchen9/job_.  This location will be purged at the end of the job.

Usage

This NVMe-based storage is excellent for I/O-intensive workloads and can be beneficial for both small and large scratch files generated on a per-task basis.
Please move any needed data to a more permanent storage location before the job ends and clear out the node local scratch.

Compilers and Software

The compiler collections are accessible through modules.  Please note that all compiles must be done on a comude nodes with APUs. Ccompilers can be loaded by executing the following command at the Linux prompt:

$ module load PrgEnv-<name>
where <name> is the name of the compiler suite. On Cosmos, PrgEnv-cray and the associated compilers and scientific libraries are loaded by default. There are 3 compiler environments available on Cosmos.

Compilers and Software
Name

Module collection

Description

CCE

PrgEnv-cray

Cray Compiling Environment

AMD

PrgEnv-amd

AMD ROCm compilers

GNU

PrgEnv-gnu

GNU Compiler Collection

The compiler suites can be swapped as a whole (so all relevant library modules will be automatically switched). For example:

$ module swap PrgEnv-cray PrgEnv-gnu

The module collection provides wrappers to the C, C++ and Fortran compilers. The commands used to invoke these wrappers are listed below.

cc: C compiler

CC: C++ compiler

ftn: Fortran compiler

No matter which vendor's compiler module is loaded, always use one of the above commands to invoke the compiler in your configures and builds. Using these wrappers will invoke the underlying compiler according to the compiler suite that is loaded in the environment.

Compiler Options

The following flags are a good starting point to achieve good performance:

Compiler Options
Compilers

Good Performance

Aggressive Optimizations

Cray C/C++

-O2 -funroll-loops -ffast-math

-Ofast -funroll-loops

Cray Fortran

Default

-O3 -hfp3

GCC

-O2 -ftree-vectorize -funroll-loops -ffast-math

-Ofast -funroll-loops

 

Detailed information about the available compiler options is available here:

Cray Compiling Environment
The GNU Compilers
The man pages of the wrappers and of the underlying compilers are also provide useful information (using the “man” command to get this information).

Singularity
Cosmos uses Singularity to support containerized software stacks. Singularity leverages a workflow and security model that makes it a very reasonable candidate for shared or multi-tenant HPC resources like the Cosmos cluster without requiring any modifications to the scheduler or system architecture. AMD provides many performant containers on the AMD Infinity Hub (https://www.amd.com/en/developer/resources/infinity-hub.html)
