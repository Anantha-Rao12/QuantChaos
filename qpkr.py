import numpy as np

class QPKR():

    def __init__(self, basis_size:int, total_time:int, params:list):
        """
        Author: Anantha Rao
        Date: 16 July 2022
        Github id: @Anantha-Rao12
        Desc: A python script to simulate the time-evolution of a quasi-periodic kicked rotor and
        determine the Anderson / metal-insulator transition in 3-dimensions
        
        Creates a Quasi-periodic kicked rotor instance with initial parameters where
        H = p^2/2 + K(t)cos(x) \sum_n \delta(t-n)
        where K(t) = K0*(1+\eps \cos(w2t + phi2) \cos(w3t + phi3))
        
        Args:
            basis_size : int that describes the size of the position and momentum basis
            total_time : total time of time evolution
            Params : [K0, hbar, eps, w2, w3, phi2, phi3] where
            K0: Kicking strength
            hbar : effective planck's constant
            eps : factor that modulates the additional temporal frequencies
            w2, w3 : the two incommensurate frequencies
            phi2, phi3 : initial phase factors


        References: 
        1. https://arxiv.org/pdf/0904.2324.pdf
        2. https://chaos.if.uj.edu.pl/~delande/Lectures/?quasiperiodic-kicked-rotor,63
        3. http://phome.postech.ac.kr/user/ams/workshop_data/loc2011delande.pdf
        4. https://verga.cpt.univ-mrs.fr/pages/kicked.html"""
    
        self.basis_size = basis_size
        self.total_time = total_time
        self.K0 = params[0]
        self.hbar = params[1]
        self.eps = params[2]
        self.w2 = params[3]
        self.w3 = params[4]
        self.phi2 = params[5]
        self.phi3 = params[6]
        self.p = np.fft.fftfreq(basis_size, 1.0/basis_size)
        self.x = np.arange(0, 2*np.pi, 2*np.pi/basis_size)
        self.Up = np.exp(-1j*(self.hbar)*(self.p**2)/2)

    def get_zerostate(self) -> np.ndarray:
        """Creates a initial state in momentum state at rest"""
        self.psi0 = np.zeros((self.basis_size), dtype=complex)
        #self.psi0[self.basis_size//2] = 1
        self.psi0[0] = 1
        return self.psi0

    def get_kickingstrength(self, time:int) -> float:
        """Provides the different kicking strengths at various times given by
        K(t) = K0*(1+\eps \cos(w2t + phi2) \cos(w3t + phi3))"""
        return self.K0*(1+self.eps*np.cos(self.w3*time+
                                           self.phi2)*np.cos(self.w2*time+
                                                        self.phi3))/self.hbar

    def get_posiitonspace_unitary(self, time:int) -> np.ndarray:
        """Provides the unitary operator in position space that is used for 
        time evolution through fourier transforms"""
        return np.exp(-1j*self.get_kickingstrength(time)*np.cos(self.x))

    def evolve(self):
        """Performs time-evolution until self.total_time and returns the average momentum^2
        or a proxy for the energy of the system with time"""
        p2 = np.zeros((self.total_time))
        psi_vectors = np.zeros((self.basis_size, self.total_time), dtype=complex)
        psi_t = self.get_zerostate()

        for t in range(self.total_time):
            Ux = self.get_posiitonspace_unitary(t)
            psi_t = np.fft.fft(Ux * np.fft.ifft (self.Up * psi_t))
            psi_vectors[:, t] = psi_t
            p2[t] = np.sum((self.p)**2 * (np.abs(psi_t)**2))
        return p2, psi_t

    def get_avg_overphases(self, no_initconfigs:int) -> np.ndarray:
        """Get time-evolved energy averaged over multiple initial configurations
        (phi2, phi3)"""
        phases1 = np.random.rand(no_initconfigs)
        phases2 = np.random.rand(no_initconfigs)
        all_p2 = np.zeros((no_initconfigs, self.total_time), dtype=np.float32)

        for idx, (phase1, phase2) in enumerate(zip(phases1, phases2)):
            self.phi2 = phase1; self.phi3 = phase2
            all_p2[idx, :], _ = self.evolve()

        return np.mean(all_p2, axis=0)

