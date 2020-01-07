CODEDIR = ../..

test: $(CODEDIR)/singularity/*.sif
	cd $(CODEDIR) && \
	singularity run --nv singularity/*.sif \
	./launch.sh 'python3.7 singularity/test/singularity_sanity_check.py'

