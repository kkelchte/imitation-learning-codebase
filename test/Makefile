CODEDIR = ..

test-cuda: $(CODEDIR)/singularity/*.sif
	singularity run --nv $(CODEDIR)/singularity/*.sif python3.7 singularity_sanity_check.py
