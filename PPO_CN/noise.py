import numpy as np
import torch

class ColoredNoiseProcess:
    """
    生成彩色噪声（Pink/Brown Noise）- 无 Scipy 依赖版本
    """
    def __init__(self, beta, size, rng=None, scale=1.0):
        """
        Args:
            beta: 噪声颜色参数 (0=White, 1=Pink, 2=Brown)
            size: 噪声张量形状 (batch_size, action_dim, seq_len)
            rng: 随机数生成器
            scale: 噪声缩放系数
        """
        self.beta = beta
        self.size = size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.scale = scale
        self.seq_len = size[-1]
        self.idx = 0
        self.sequence = None
        self._generate_sequence()

    def _generate_sequence(self):
        """生成完整的彩色噪声序列"""
        if self.beta <= 0.0:
            # 白噪声（标准高斯）
            self.sequence = self.rng.standard_normal(self.size) * self.scale
        elif abs(self.beta - 2.0) < 0.5:
            # Brown Noise (Integrate White Noise)
            # x[t] = x[t-1] + white_noise
            white = self.rng.standard_normal(self.size)
            # Cumulative sum along time axis (last axis)
            brown = np.cumsum(white, axis=-1)
            # Normalize to unit variance (approximately) to match scale
            # Brown noise variance grows with time, so we normalize the whole sequence
            std = np.std(brown, axis=-1, keepdims=True)
            self.sequence = (brown / (std + 1e-8)) * self.scale
        else:
            # Pink Noise (1/f) - Voss-McCartney Algorithm approximation
            # Or simple IIR approximation: x[t] = 0.9 * x[t-1] + 0.1 * white
            # Let's use a simplified 1/f generation via FFT if numpy allows,
            # otherwise simple autoregressive for approximation.

            # Using FFT method (NumPy has fft)
            # Generate white noise in frequency domain
            num_samples = self.seq_len
            # Construct scaling factors
            f = np.fft.fftfreq(num_samples)
            f[0] = 1e-8 # Avoid division by zero

            # 1/f^beta power spectrum -> 1/f^(beta/2) amplitude spectrum
            scaling = 1 / np.power(np.abs(f), self.beta / 2.0)
            scaling[0] = 0 # Remove DC component

            # Generate for each batch/action element
            # To be efficient, we reshape to (-1, seq_len)
            flat_size = (int(np.prod(self.size[:-1])), num_samples)
            white_spec = np.fft.fft(self.rng.standard_normal(flat_size))

            pink_spec = white_spec * scaling
            pink = np.real(np.fft.ifft(pink_spec))

            # Normalize
            std = np.std(pink, axis=-1, keepdims=True)
            pink = (pink / (std + 1e-8)) * self.scale

            self.sequence = pink.reshape(self.size)

        self.idx = 0

    def sample(self):
        """采样一个时间步的噪声"""
        if self.sequence is None or self.idx >= self.seq_len:
            self._generate_sequence()

        # 取出当前时间步的噪声: [batch_size, action_dim]
        noise = self.sequence[..., self.idx]
        self.idx += 1
        return noise

class ColoredNoiseExploration:
    """
    为 PPO 的 Logits 添加彩色噪声
    """
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
        """
        Args:
            logits: [batch_size, num_actions]
        Returns:
            noisy_logits: [batch_size, num_actions]
        """
        # 获取 numpy 格式的噪声
        noise = self.noise_process.sample() # [batch_size, num_actions]

        # 转为 Tensor
        if self.device is None:
            self.device = logits.device

        noise_tensor = torch.tensor(noise, dtype=logits.dtype, device=self.device)

        # 确保 batch size 匹配
        if noise_tensor.shape[0] != logits.shape[0]:
             noise_tensor = noise_tensor[:logits.shape[0]]

        return logits + noise_tensor
