# molecule_adding
Forked from Jens Vive Kildgaard.
https://github.com/JensVK/molecule_adding

Systematic addition of a small molecule to an existing cluster/molecule. It is discussed in detail in this paper:

"Hydration of Atmospheric Molecular Clusters: A New Method for Systematic Configurational Sampling"<br>
Jens Vive Kildgaard, Kurt V. Mikkelsen, Merete Bilde, and Jonas Elm,
*J. Phys. Chem. A* **2018**, 122, 5026−5036
http://doi.org/10.1021/acs.jpca.8b02758

Overall, it's a very nice and convenient tool I hope to play around with more.

# Typical use

`python ../molecule_adding.py --fibonacci_points 1 --output dimer --header header.txt --format ".xyz" solute.xyz solvent.xyz`

* fibonacci_points is 5 by default, but it was creating a much larger number of isomers than
  necessary for my test case
* including a header file with number of atoms and a comment line to comply with Cartesian XYZ format helps.


Citation
--------

Please cite doi.org/10.1021/acs.jpca.8b02758 if using it for scientific publications.
