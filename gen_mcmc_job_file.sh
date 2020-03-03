#job file generation for inverse problem
RESOLUTION=64
BETA=50
LENGTH_SCALE_X=0.3
LENGTH_SCALE_Y=0.1
PERM_VARIANCE=2.0
NCORES=8
NSAMPLES=2000
NWARMUP=500

NAMEBASE="inverse_problem_fom_mcmc"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/rom_inverse"
JOBNAME="${DATESTR}_inverse_mcmc_resolution=${RESOLUTION}"
JOBDIR="/home/constantin/python/data/rom_inverse/mcmc/resolution=${RESOLUTION}/beta=${BETA}/l=${LENGTH_SCALE_X}_${LENGTH_SCALE_Y}/var=${PERM_VARIANCE}/n_samples=${NSAMPLES}/n_warmup=${NWARMUP}/${DATESTR}"


#Create job directory and copy source code
mkdir -p "${JOBDIR}"
cp -r "$PROJECTDIR/poisson_fem" $JOBDIR
cp "$PROJECTDIR/ROM.py" $JOBDIR
cp "$PROJECTDIR/rom_inverse.py" $JOBDIR
cp "$PROJECTDIR/rom_inverse_mcmc.py" $JOBDIR


#Change directory to job directory; completely independent from project directory
cd "$JOBDIR"
echo "Current directory:"
echo $PWD


#construct job file string
echo "#!/bin/bash" >> ./job_file.sh
echo "#SBATCH --job-name=${JOBNAME}" >> ./job_file.sh
echo "#SBATCH --partition batch_SKL,batch_SNB" >> ./job_file.sh
echo "#SBATCH --nodes 1-1" >> ./job_file.sh
echo "#SBATCH --mincpus=${NCORES}" >> ./job_file.sh     #node is not shared with other jobs
echo "#SBATCH --output=/home/constantin/OEfiles/${JOBNAME}.%j.out" >> ./job_file.sh
echo "#SBATCH --error=/home/constantin/OEfiles/${JOBNAME}.%j.err" >> ./job_file.sh
echo "#SBATCH --mail-type=ALL" >> ./job_file.sh
echo "#SBATCH --mail-user=mailscluster@gmail.com " >> ./job_file.sh
echo "#SBATCH --time=240:00:00" >> ./job_file.sh

echo "" >> ./job_file.sh
echo "#Switch to job directory" >> ./job_file.sh
echo "cd \"$JOBDIR\"" >> ./job_file.sh
echo "" >> ./job_file.sh
echo "#Set parameters" >> ./job_file.sh
echo "sed -i \"24s/.*/print('inverse temperature beta == ', beta := ${BETA})/\" ./rom_inverse_mcmc.py" >> ./job_file.sh
echo "sed -i \"25s/.*/print('fom resolution == ', lin_dim_fom := ${RESOLUTION}, ' x ', lin_dim_fom)/\" ./rom_inverse_mcmc.py" >> ./job_file.sh
echo "sed -i \"27s/.*/permeability_length_scale = torch.tensor([${LENGTH_SCALE_X}, ${LENGTH_SCALE_Y}])/\" ./rom_inverse_mcmc.py" >> ./job_file.sh
echo "sed -i \"28s/.*/permeability_variance = torch.tensor(${PERM_VARIANCE})/\" ./rom_inverse_mcmc.py" >> ./job_file.sh
echo "sed -i \"29s/.*/n_chains = ${NCORES}/\" ./rom_inverse_mcmc.py" >> ./job_file.sh
echo "sed -i \"30s/.*/n_samples = ${NSAMPLES}/\" ./rom_inverse_mcmc.py" >> ./job_file.sh
echo "sed -i \"31s/.*/n_warmup = ${NWARMUP}/\" ./rom_inverse_mcmc.py" >> ./job_file.sh

#Activate rom_inverse environment and run python
echo "source ~/.bashrc" >> ./job_file.sh
echo "conda activate rom_inverse" >> ./job_file.sh
echo "python -O ./rom_inverse_mcmc.py" >> ./job_file.sh
echo "" >> ./job_file.sh


chmod +x job_file.sh
#directly submit job file
sbatch job_file.sh
#./job_file.sh	#to test in shell









