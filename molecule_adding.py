#!/usr/bin/python env


import numpy as np
import os
import argparse
import itertools
from scipy.spatial import distance
from string import ascii_lowercase

try:
    from mendeleev import element
    vdw_radii = True
except:
    print("The python package mendeleev was not found: https://pypi.python.org/pypi/mendeleev\n van der Waals radii set to 1.2") 
    vdw_radii = False

  
def centroidize(coordinates):
    """
    Reorinatates the solvent molecule to centroid
    """
    sum_x = np.sum(coordinates[:,0])
    sum_y = np.sum(coordinates[:,1])
    sum_z = np.sum(coordinates[:,2])
    l = len(coordinates)
    centroid = np.array( [ sum_x/float(l), sum_y/float(l), sum_z/float(l)] )
    return coordinates - centroid
    
def radius(atom):
    if vdw_radii:
        # Element return a pm radius
        return element(atom).vdw_radius * 10**-2
    else:
        return 1.2
        
def max_vdw(coordinates, radii):
    cc = centroidize(coordinates)
    return max( np.linalg.norm(cc, axis=1) + radii)

def rotation_matrix(theta_x,theta_y,theta_z):
    """Rotation matrix
    """
    return np.array([[np.cos(theta_y)*np.cos(theta_z),
             -np.cos(theta_x)*np.sin(theta_z) + np.sin(theta_x)*np.sin(theta_y)*np.cos(theta_z),
             np.sin(theta_x)*np.sin(theta_z) + np.cos(theta_x)*np.sin(theta_y)*np.cos(theta_z)],
            [np.cos(theta_y)*np.sin(theta_z),
             np.cos(theta_x)*np.cos(theta_z) + np.sin(theta_x)*np.sin(theta_y)*np.sin(theta_z),
            -np.sin(theta_x)*np.cos(theta_z) + np.cos(theta_x)*np.sin(theta_y)*np.sin(theta_z)],
            [-np.sin(theta_y), np.sin(theta_x)*np.cos(theta_y),np.cos(theta_x)*np.cos(theta_y)]])

def rotate_molecule(molecule,x,y,z):
    matrix = rotation_matrix(x,y,z)
    #Coordinates are given in Nx3 matrix, therefore the reverse order and transpose.
    return np.dot(molecule,matrix.T)

def solvent_rotated(solvent,rotation_n):
    solvent_rotated = np.empty([rotation_n**3,len(solvent),3])
    count = 0
    for x in np.linspace(0,2*np.pi,rotation_n, endpoint=False):
        for y in np.linspace(0,2*np.pi,rotation_n, endpoint=False):
            for z in np.linspace(0,2*np.pi,rotation_n, endpoint=False):
                solvent_rotated[count,:,:] = rotate_molecule(solvent,x,y,z)
                count += 1
    return solvent_rotated
    
def solvent_alignment_matrix(R,r):
    #r is the water molecule OH vector and R is the OH in the molecule that water should align
    R = R * np.linalg.norm(R)**-1
    r = r * np.linalg.norm(r)**-1
    return np.eye(len(R))* np.dot(r,R) + np.linalg.norm(np.cross(r,R)) * (np.tensordot(R,r,axes=0) - np.tensordot(r,R,axes=0)) + ( 1 + np.dot(r,R)) * np.tensordot(np.cross(r,R),np.cross(r,R),axes=0)

def rotation_around_vector_matrix(vector,theta):
    vector = np.linalg.norm(vector)**-1 * vector
    x , y , z = [i for i in vector]
    return np.array([[np.cos(theta) + x**2*(1-np.cos(theta)) , x*y*(1-np.cos(theta)) - z*np.sin(theta) , x*z*(1-np.cos(theta)) + y*np.sin(theta)],
                      [y*x*(1-np.cos(theta)) + z*np.sin(theta) , np.cos(theta) + y**2*(1-np.cos(theta)) , y*z*(1 - np.cos(theta)) - x*np.sin(theta)],
                       [ z*x*(1-np.cos(theta)) - y*np.sin(theta) , z*y*(1-np.cos(theta)) + x*np.sin(theta) , np.cos(theta) + z**2*(1 - np.cos(theta))]])

def re_center_solvent(solvent_a,solvent_c):
    """
    Recenter the solvent molecule, most likely water, to be centered at the oxygen
    """
    H_list, O_list = get_H_O_index_list(solvent_a)
    return H_list, O_list, solvent_c - solvent_c[O_list[0]]
            
    
def get_coordinates(filename):
    #genfromtxt skips the first 2 lines and then make numpy array from the rest
    xyz = np.genfromtxt(str(filename),dtype=str,skip_header=2)
    #Returns atoms in array as strings and array of coordinate [[x,y,z],...] as floats
    return xyz[0:,0:1] , xyz[0:,1:].astype(np.float)

def write_molecules(atoms, coordinates, filename, fmt, header, footer, suffix):
    number_of_structures = len(coordinates)
    if suffix and suffix.isalpha():
        #Find number of letters needed.
        n_suffix = 0
        while True:
            if 26**n_suffix > number_of_structures:
                break
            else: n_suffix += 1
        suffix_list = iter_all_strings(n_suffix)
    elif not suffix or suffix.isdigit():
        n_suffix = len(str(number_of_structures))
        suffix_list = [ str(i).zfill(n_suffix) for i in range(number_of_structures)]
    for i, structure in enumerate(coordinates):        
        with open(str(filename) + '_' + str(suffix_list[i]) + str(fmt), 'w' ) as f:
            if header:
                for line in open(header):
                    f.write(line)
            for x in range(len(atoms)):
                f.write("{0:2s} {1:15.8f} {2:15.8f} {3:15.8f}\n".format(atoms[x,0], structure[x, 0], structure[x, 1], structure[x, 2]))
            if footer:
                for line in open(footer):
                    f.write(line)
            f.close()
    return
    
def iter_all_strings(size):
    # Creates a list for alphabetic ordering the output files
    return [ ''.join(i) for i in itertools.product(ascii_lowercase, repeat=size)]

def combine(solute_a,solute,solvent_a,solvent):
    # Combine the solute molecule and solvent molecule
    atoms = np.concatenate((solute_a,solvent_a))    
    coordinates = np.concatenate((solute,solvent))
    return atoms , coordinates
    
def make_sphere(R,n):
    """Fibonacci sphere (evenly distribution of n points)
    """
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1. - 1.0 / n, 1.0 / n - 1., n)
    radius = np.sqrt(1 - z * z)
    
    points = np.zeros((n, 3))
    points[:,0] = R*radius * np.cos(theta)
    points[:,1] = R*radius * np.sin(theta)
    points[:,2] = R*z
    return points
    
def solute_sphere(solute,radii,n):
    """
    Creates a list of points around the solute molecule using Fibonacci spheres around each atom
    """
    solute_sphere = []
    for atom_c, radius in zip(solute,radii):
        fibo_sphere = make_sphere(radius,n)
        for sphere_point in fibo_sphere:
            solute_sphere_point = atom_c + sphere_point
            if all(np.linalg.norm(solute_sphere_point - a) >= 0.9*r for a, r in zip(solute,radii)):
                solute_sphere.append(solute_sphere_point)
    return np.asarray(solute_sphere)
        
def get_H_O_index_list(atoms):
    """Returns two lists with the indices of hydrogen and oxygen atoms.
    """
    H_list = O_list = []
    for index, atom in enumerate(atoms):
        if atom[0] == 'H':
            H_list.append(index)
        if atom[0] == 'O':
            O_list.append(index)
    return H_list, O_list

def perturb_molecule(molecule_c,point,length,sigma):
    #Create a set of normelized vectors from the midpoint to all atoms in the solute (returns lists with x, y and z compondends (transpose to get xyz coordinates))
    norm_molecule_midpoint = np.subtract(molecule_c,point).T * (np.linalg.norm(np.subtract(molecule_c,point) , axis=1))**-1
    #Calculate how much each atom shall be moved depending on their distance to the midpoint in a gauss dist (returns a list N_solute_atom long)
    gaussian_dist = np.exp(-np.linalg.norm(molecule_c - point,axis=1)**2/sigma**2)
    #To get the OH length to be 2. Aa to fit water inbetween multiply by
    factor = 0.5 * (2.2 - length) * np.exp( length**2 * ( 4 * sigma**2)**-1 )
    #Multiply all to strech the molecule/solute "away" from the midpoint and add the solute molecule coordinates (transpose for xxx yyy zzz to xyz xyz xyz)
    return (norm_molecule_midpoint * gaussian_dist * factor).T + molecule_c
    
def solvent_placement(solvent_a,solvent_c,point,vector,theta):
    oxygen , hydrogen , recentered_solvent = re_center_solvent(solvent_a,solvent_c)
    #Use the first hydrogen and oxygen in the water molecule to define the OH to align
    rotated_solvent = np.dot(recentered_solvent,solvent_alignment_matrix(vector,recentered_solvent[hydrogen[0]]-recentered_solvent[oxygen[0]]).T)
    schrinked_solvent = min(distance.pdist(rotated_solvent))**-1 * 0.65 * rotated_solvent
    schrinked_solvent = np.dot(schrinked_solvent , rotation_around_vector_matrix(vector,theta).T)
    placed_solvent = schrinked_solvent + point + 0.5 * (schrinked_solvent[hydrogen[0]]-schrinked_solvent[oxygen[0]])
    return placed_solvent
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('solute_filename', metavar='solute.xyz', type=str)
    parser.add_argument('solvent_filename', metavar='solvent.xyz', type=str)
    parser.add_argument("-o", "--output", help="Specify the name of the output file.", type=str)
    parser.add_argument("-t", "--header", help="Add a file as a header to the output file.", type=str)
    parser.add_argument("-f", "--footer", help="Add a file as a footer to the output file.", type=str)
    parser.add_argument("-b", "--break_hydrogen_bonds", help="Specify if existing hydrogen bonds should be broken and solvent molecule placed in between. Implemented specific for water.", action='store_true' )
    parser.add_argument("--format", help="Specify the output file extension including the .", type=str, default="" )
    parser.add_argument("-s", "--suffix", help="Suffix type indicated by either a letter or a number. Default numcerial")
    parser.add_argument("-d", "--directory", help="Directory where the output files are placed." , type=str)
    parser.add_argument("-n", "--fibonacci_points", help="Number of points around each atom", type=int, default=5)
    parser.add_argument("-R", "--rotation_orientation", help="Specify the splitting of the rotation around the x,y and z axis.", type=int, default=3)
    parser.add_argument("-r", "--rotation_hydrogen_bond", help="The number of different orientation the solvent should have in the former hydrogen bond.", type=int, default=3)
    parser.add_argument("--sigma", help="Set sigma - the perturbation length.", type=float, default=2.5)
    args = parser.parse_args()

    
    solvent_atoms , solvent_c = get_coordinates(args.solvent_filename)
    solute_atoms , solute_c = get_coordinates(args.solute_filename)
    
    solvent_radii = [ radius(atom) for atom in solvent_atoms[:,0]]
    solute_radii = [ radius(atom) for atom in solute_atoms[:,0]]
    
    print(args)
    coordinates = []
    for point in solute_sphere(solute_c,solute_radii,args.fibonacci_points):
        for r_solv in solvent_rotated(solvent_c, args.rotation_orientation):
            new_solvent = point + r_solv # previuously np.add(point,r_solv)
            distance_list = distance.cdist(solute_c,new_solvent).flatten()
            if all(i > 0.6 for i in distance_list):
                coordinates.append(np.concatenate((solute_c,new_solvent)))
            else:
                schrinked_solvent = np.add(1.1*point,0.9*r_solv)
                distance_list = distance.cdist(1.1*solute_c,schrinked_solvent).flatten()
                if all(i > 0.6 for i in distance_list):
                    coordinates.append(np.concatenate((1.1*solute_c,schrinked_solvent)))
    if args.break_hydrogen_bonds:
        H_list, O_list = get_H_O_index_list(solute_atoms)
        for h in H_list:
            for o in O_list:
                OH_length = np.linalg.norm(solute_c[h]-solute_c[o])
                if OH_length < 2. and OH_length > 1.1:
                    OH_midpoint = 0.5 * ( solute_c[h] + solute_c[o] ) 
                    # check if any atoms are closer than the oxygen and hydrogen responsiple for the OH bond, this can be the case for peroxides.
                    if any(i < 0.5*OH_length for i in distance.cdist(solute_c , [OH_midpoint])): 
                        break
                    perturb_solute = perturb_molecule(solute_c,OH_midpoint,OH_length, args.sigma)
                    R = perturb_solute[h] - perturb_solute[o] 
                    OH_midpoint = 0.5 * ( perturb_solute[h] + perturb_solute[o] ) 
                    n_theta = args.rotation_hydrogen_bond
                    start = 0
                    while True:
                        count = 0
                        solvent_list = []
                        for theta in np.linspace(start,2*np.pi + start,n_theta,endpoint=False):
                            solvent_placed = solvent_placement(solvent_atoms,solvent_c,OH_midpoint,R,theta)
                            solvent_list.append(solvent_placed)
                            distance_list = distance.cdist(perturb_solute , solvent_placed).flatten()
                            if all(i > 0.51 for i in distance_list):
                                count += 1
                                continue
                            else: break
                        if count == n_theta:
                            for solv in solvent_list:
                                combined_a , combined_c = combine(solute_atoms,perturb_solute,solvent_atoms,solv)
                                coordinates.append(np.concatenate((perturb_solute , solv)) )
                            break
                        start += np.pi * 10**-1
                        if start == 2*np.pi:
                            start = 0
                            n_theta -= 1
                        if n_theta == 0:
                            break
    
    if args.output:
        filename = args.output
    else:
        #Use the solute's filename as default without extension
        filename = args.solute_filename.split('.',1)[0]
    if args.directory:
        try: 
            os.makedirs(args.directory)
        except:
            pass #already exists
        filename = os.path.join(args.directory, filename)
    write_molecules(np.concatenate((solute_atoms, solvent_atoms)), coordinates, filename, args.format, args.header, args.footer, args.suffix)
    



