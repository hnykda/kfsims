# kfsims
This is an implementation of Diffusion Variational Bayes Adaptive Kalman Filter 
by me - Daniel Hnyk - and it serves as a basis for simulations in my Master's
thesis. 

# Reproducible research
As long as you install the same libraries, the simulations in `nbs` directory
should give you the same results and should be directly executable. 

Please report any issues or bugs here in GitHub. PRs are, of course, welcome. 


# Installation
Clone the repository and just run `pip install -e .`. It should install all the
dependencies and then you can open jupyter notebooks in `nbs` folder where
is the summary of all simulations we did. It contains all the charts used
in the thesis. 

There is then one dependency not covered in the `setup.py`'s `install_requires`
and that is Jupyter Notebook, which is not mandatory to run the logic of the code
but you won't be able to reran the simulations as we did in the work. 

Many thanks to my supervisor Kamil Dedecius.      
