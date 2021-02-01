import numpy as np
from typing import Optional


class KickedRotor1D():
    
    def __init__(self,kicking_strength: float,initial_state: Optional[np.ndarray] = None) -> None:
        
        if initial_state == None:
            initial_state = np.random.rand(2,1)
            
        elif len(initial_state) != 2:
            raise ValueError('Incorrect initial state. Expecting a list of length 2')
            
        elif kicking_strength < 0:
            raise ValueError('Incorrect Kicjign strength. Value cannot be negative')
            
        self.initial_state = np.array(initial_state).reshape(2,1)
        self.kicking_strength = kicking_strength
        
    def chirikov_map(self,phasepoint:np.ndarray,kicking_strength:float) -> np.ndarray:
        """Returns the Chirikov map of the given phase-point"""

        new_phasepoint = np.zeros((phasepoint.shape))
        new_phasepoint[1] = np.mod(phasepoint[1] + kicking_strength*np.sin(phasepoint[0]),2*np.pi)
        new_phasepoint[0] = np.mod((phasepoint[0] + new_phasepoint[1]),2*np.pi)
        return new_phasepoint
    
    def get_phasespace(self, ntimesteps: int) -> np.ndarray:
        """Returns the phasespace vector for the given number of timesteps"""
        
        phasespace = np.zeros((2,ntimesteps))
        phasespace[:,0:1] = self.initial_state
        for i in range(len(phasespace[0])-1):
            phasespace[:,i+1] = self.chirikov_map(phasespace[:,i],self.kicking_strength)
        return phasespace 