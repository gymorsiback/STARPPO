import numpy as np
import torch

class ColoredNoiseProcess:
    def __init__(self, beta, size, rng=None, scale=1.0):
        self.beta = beta
        self.size = size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.scale = scale
        self.seq_len = size[-1]
        self.idx = 0
        self.sequence = None
        self._generate_sequence()
    
    def _generate_sequence(self):
        if self.beta <= 0.0:
            self.sequence = self.rng.standard_normal(self.size) * self.scale
        elif abs(self.beta - 2.0) < 0.5:
            white = self.rng.standard_normal(self.size)
            brown = np.cumsum(white, axis=-1)
            std = np.std(brown, axis=-1, keepdims=True)
            self.sequence = (brown / (std + 1e-8)) * self.scale
        else:
            num_samples = self.seq_len
            f = np.fft.fftfreq(num_samples)
            f[0] = 1e-8 
            scaling = 1 / np.power(np.abs(f), self.beta / 2.0)
            scaling[0] = 0 
            flat_size = (int(np.prod(self.size[:-1])), num_samples)
            white_spec = np.fft.fft(self.rng.standard_normal(flat_size))
            
            pink_spec = white_spec * scaling
            pink = np.real(np.fft.ifft(pink_spec))
    
            std = np.std(pink, axis=-1, keepdims=True)
            pink = (pink / (std + 1e-8)) * self.scale
            
            self.sequence = pink.reshape(self.size)
        
        self.idx = 0
    
    def sample(self):
        if self.sequence is None or self.idx >= self.seq_len:
            self._generate_sequence()

        noise = self.sequence[..., self.idx]
        self.idx += 1
        return noise

class ColoredNoiseExploration:
    def __init__(self, beta, num_envs, num_actions, seq_len=2048, scale=0.1, rng=None):
        self.beta = beta
        self.device = None 
        self.noise_process = ColoredNoiseProcess(
            beta=beta, 
            size=(num_envs, num_actions, seq_len),
            rng=rng,
            scale=scale
        )
    
    def add_noise_to_logits(self, logits):
        noise = self.noise_process.sample() 

        if self.device is None:
            self.device = logits.device
            
        noise_tensor = torch.tensor(noise, dtype=logits.dtype, device=self.device)

        if noise_tensor.shape[0] != logits.shape[0]:
             noise_tensor = noise_tensor[:logits.shape[0]]
             
        return logits + noise_tensor
