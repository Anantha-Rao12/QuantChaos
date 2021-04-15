import numpy as np
from typing import Optional


class KickedRotor1D():
    """Implements a classical kicked rotor object with initial state and kicking strength"""
    
    def __init__(self,kicking_strength: float,initial_state: Optional[np.ndarray] = None) -> None:
        
        if initial_state == None:
            initial_state = np.random.rand(2,1)
            
        elif len(initial_state) != 2:
            raise ValueError('Incorrect initial state. Expecting a state of length 2')
            
        elif kicking_strength < 0:
            raise ValueError('Incorrect Kicking strength. Value cannot be negative')
            
        self.initial_state = np.array(initial_state).reshape(2,1)
        self.kicking_strength = kicking_strength
        
    def chirikov_map(self,phasepoint:np.ndarray,kicking_strength:float,keep_modulus) -> np.ndarray:
        """Returns the Chirikov map of the given phase-point
        Args:
            phasepoint: 2-dim vector specifying initial state
            kicking_strength: float value specifying the strength of periodic force
            keep_modulus: Boolean used to divide dynamical variables by 2pi"""

        new_phasepoint = np.zeros((phasepoint.shape))
        if keep_modulus == True:
            new_phasepoint[1] = np.mod(phasepoint[1] + kicking_strength*np.sin(phasepoint[0]),2*np.pi)
            new_phasepoint[0] = np.mod((phasepoint[0] + new_phasepoint[1]),2*np.pi)
        else:
            new_phasepoint[1] = phasepoint[1] + kicking_strength*np.sin(phasepoint[0])
            new_phasepoint[0] = phasepoint[0] + new_phasepoint[1]
        return new_phasepoint
    
    def get_phasespace(self, ntimesteps: int,keep_modulus=True) -> np.ndarray:
        """Returns the phasespace vector for the given number of timesteps by iteratively computing the chirikov map"""
        
        phasespace = np.zeros((2,ntimesteps))
        phasespace[:,0:1] = self.initial_state
        for i in range(len(phasespace[0])-1):
            phasespace[:,i+1] = self.chirikov_map(phasespace[:,i],self.kicking_strength,keep_modulus)
        return phasespace 
    
    def get_diffusion(self, ntimesteps: int,keep_modulus=False):
        """Return p^2/2K for the given number of timesteps
        Args:
            ntimesteps: No of time steps for time evolution
            keep_modulus: Boolean used to divide dynamical variables by 2pi"""
        phasespace = self.get_phasespace(ntimesteps,keep_modulus)
        return (phasespace[0,:]**2)/self.kicking_strength