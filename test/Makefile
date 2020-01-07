CODEDIR = ..

test-cuda: $(CODEDIR)/singularity/*.sif
	singularity run --nv $(CODEDIR)/singularity/*.sif ${COMMAND}
