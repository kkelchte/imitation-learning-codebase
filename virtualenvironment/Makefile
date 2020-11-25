.PHONY: create_env

install:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	/bin/sh Miniconda3-latest-Linux-x86_64.sh && \
	rm Miniconda3-latest-Linux-x86_64.sh

create_env: install
	conda create --name venv python=3.8 && \
	conda activate venv && \
	conda install --yes --file requirements-conda && \
	conda install --yes pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch && \
	python3 -m pip install -r requirements-pip
