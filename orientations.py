import numpy as np
import matplotlib.pyplot as plt
from quat import Quat
from sys import exit

def ori_matrix(phi1,Phi,phi2,passive=True):

    '''
    Returns (passive) orientation matrix, as a np.matrix from
    3 euler angles (in degrees). 

    '''
    
    phi1=np.radians(phi1)
    Phi=np.radians(Phi)
    phi2=np.radians(phi2)
    R11 = np.cos(phi1)*np.cos(phi2)-np.sin(phi1)*np.cos(Phi)*np.sin(phi2)
    R12 = np.sin(phi1)*np.cos(phi2)+ np.cos(phi1)*np.cos(Phi)*np.sin(phi2)
    R13 = np.sin(phi2)*np.sin(Phi)
    R21 = -np.cos(phi1)*np.sin(phi2)-np.sin(phi1)*np.cos(Phi)*np.cos(phi2)
    R22 = - np.sin(phi1)*np.sin(phi2)+np.cos(phi1)*np.cos(Phi)*np.cos(phi2)
    R23 = np.cos(phi2)*np.sin(Phi)
    R31 = np.sin(phi1)*np.sin(Phi)
    R32= -np.sin(Phi)*np.cos(phi1)
    R33= np.cos(Phi)
    matrix=np.matrix([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])
    if not passive: # matrix above is for the passive rotation 
        matrix=matrix.transpose()
    return matrix

def get_proj(g,pole,proj='stereo'):
    ''' 
    Returns polar projection vector from an orientation matrix (g),
    a pole vector (pole) using either stereographic or equal area projection,
    '''
    n=np.linalg.norm(pole)
    pole=np.matrix(pole).T/n
    vector=g.T*pole #invert matrix
    alpha=np.arccos(vector[2])
    if np.isclose(alpha,0.0):
        beta=np.matrix([0.0])
    else:
        beta=np.arctan2(vector[1]/np.sin(alpha),vector[0]/np.sin(alpha))
       
    if alpha>np.pi/2:
        alpha = np.pi-alpha
        beta += np.pi
    
    if proj=='stereo':
        Op=np.tan(alpha/2)
    if proj=='equal area':
        Op=np.sqrt(2)*np.sin(alpha/2)
        
    return Op, beta,vector

def plot_poles(beta_list, Op_list):
    '''
    Plots a pole figure from a list of angles (beta_list) in radians, 
    and a list of radii (Op_list). 
    '''

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='polar')
    for beta, Op in zip(beta_list, Op_list):
        ax.plot(beta, Op,'o',ms=18, alpha=0.5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_ylim([0,1.0])

def get_vectors(g,pole_list, proj='stereo'):
    '''
    Returns a list of vectors and correponding polar angle and radius
    for an orientation matrix (g), using a chosen projection (proj).
    '''  

    Op_list=[]
    beta_list=[]
    vector_list=[]
    
    for pole in pole_list:
        Op,beta,vector=get_proj(g,pole,proj)
        Op_list.append(Op)
        beta_list.append(beta)
        vector_list.append(vector)
    
    return Op_list, beta_list, vector_list

def get_pole_list(pole='001'):
    ''' 
    Returns all symmetric variants of the '001', '110' or '111' poles.

    '''

    pole_dict = {
        '001': [np.array([1,0,0]), np.array([0,1,0]),np.array([0,0,1]),
               np.array([-1,0,0]), np.array([0,-1,0]),np.array([0,0,-1])],
        '110': [np.array([1,1,0]), np.array([1,0,1]),np.array([0,1,1]),
               np.array([-1,-1,0]), np.array([-1,0,-1]),np.array([0,-1,-1]),
               np.array([-1,1,0]), np.array([-1,0,1]),np.array([0,-1,1]),
               np.array([1,-1,0]), np.array([1,0,-1]),np.array([0,1,-1])],
        '111': [np.array([1,1,1]), np.array([-1,1,1]),np.array([1,-1,1]), np.array([1,1,-1]),
               np.array([-1,-1,-1]), np.array([-1,-1,1]),np.array([1,-1,-1]), np.array([-1,1,-1])]
    }
    pole_list=pole_dict[pole]
    return pole_list


# Utility functions

def euler_to_quat(phi1,Phi,phi2):
    """Takes euler angles in degrees and returns quaternion"""
    parent_eulers = np.radians(np.array((phi1, Phi, phi2)))
    ori_quat = Quat(*parent_eulers)
    return ori_quat

# Plotting

def plot_pole(g, pole='001', legend=False):
    ''' Plots a pole figure from an orientation matrix (g)
        for s chosen pole (pole). Pole can be the '001', '110' or '111'.
    '''
    
    pole_list=get_pole_list(pole)
    Op_list, beta_list, vector_list=get_vectors(g,pole_list)
    plot_poles(beta_list, Op_list)
    if legend:
        plt.legend(pole_list,bbox_to_anchor=(1.5, 1.05))
    #return beta_list, Op_list
    
def plot_all_poles(g, proj= 'stereo', fig = None, label='ro'):
    ''' Plots 3 pole figures from an orientation matrix (g)
        for s chosen pole (pole), correponding to the '001', '110' 
        and '111' poles.
    '''
    poles=['001','110','111']
    if fig == None:
        fig = plt.figure(figsize=(10,30))
    for n, pole in enumerate(poles):
        pole_list=get_pole_list(pole)
        Op_list, beta_list, vector_list=get_vectors(g,pole_list,proj)
        ax = fig.add_subplot(1,3,n+1, projection='polar')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim([0,1.0])
        ax.set_title(pole)
        for beta, Op in zip(beta_list, Op_list):
            ax.plot(beta, Op,label,ms=14, alpha=0.5)

def plot_all_poles_multi(g_list, proj= 'stereo', fig = None ,label='go' ):
    ''' Plots 3 pole figures from a list of orientation matrices (g_list)
        for s chosen pole (pole), correponding to the '001', '110' 
        and '111' poles.
    '''
    poles=['001','110','111']
    if fig == None:
        fig = plt.figure(figsize=(10,30))
    pole_figures=[]
    for n, pole in enumerate(poles): 
        ax = fig.add_subplot(1,3,n+1, projection='polar')
        pole_figures.append(ax)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim([0,1.0])
        ax.set_title(pole)
    for g in g_list:
        for pole_figure, pole in zip(pole_figures,poles):
            pole_list=get_pole_list(pole)
            Op_list, beta_list, vector_list=get_vectors(g,pole_list,proj)
            for beta, Op in zip(beta_list, Op_list):
                pole_figure.plot(beta, Op, label,ms=10, alpha=0.5)

def twinOrientation(parentOri, axis, angle = 60.0):
    """Calculates twin orientation quaternion from parent orientation quaternion
        by rotating about axis by angle"""
    twinRot = Quat.fromAxisAngle(axis, np.radians(angle))
    twinOri = twinRot * parentOri
    return twinOri


