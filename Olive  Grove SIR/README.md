This was my first stab at an SIR model , it is not full complete. 
It was a recreation of the results from the paper :A lattice model to manage the vector and the infection of the Xylella fastidiosa on olive trees by Annalisa Fierro, Antonella Liccardo & Francesco Porcelli.<br>
I found the Olive rapid decline issue really interesting, lots of research has been done in the southern Italian universities.
The paper also tested the effect of preventive measures, I only simulated the unaltered closed Olive grove system. 
The processes were purely stochastic using python random funcitons, rather than PDE models.<br>
I would like to try some other SIR models, but i found the lattice model a good intro, especially as I have some experience with Ising models etc. in physics. <br>
The code takes a while to run, it just tracks the position of the disease vecotrs and each interaction. I had to do some sampling so my computer could run it. <br>
There is an interesting Machine leanring program that can identify affected olive tress via statilite, that would be an interesting combination with this papers work.


<p align="center">
  <img src="Olivegrove_results.png" 
       alt="Olivegrove_results" width="950" style="margin: 10px;">
</p>
