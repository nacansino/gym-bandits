import pytest
import numpy as np

from gym_bandits.bandit import (
    BanditEnv,
    BanditTwoArmedDeterministicFixed,
    BanditTwoArmedHighLowFixed,
    BanditTwoArmedHighHighFixed,
    BanditTwoArmedLowLowFixed,
    BanditTwoArmedUniform,
    BanditTenArmedRandomFixed,
    BanditTenArmedUniformDistributedReward,
    BanditTenArmedRandomRandom,
    BanditTenArmedGaussian,
)


class TestBanditEnv:
    def test_invalid_length(self):
        with pytest.raises(ValueError, match="Probability and Reward distribution must be the same length"):
            BanditEnv(p_dist=[0.5, 0.5], r_dist=[1])

    def test_invalid_probability(self):
        with pytest.raises(ValueError, match="All probabilities must be between 0 and 1"):
            BanditEnv(p_dist=[1.5, 0.5], r_dist=[1, 1])

    def test_invalid_std(self):
        with pytest.raises(ValueError, match="Standard deviation in rewards must all be greater than 0"):
            BanditEnv(p_dist=[0.5, 0.5], r_dist=[[1, 0], [1, 1]])


class TestBanditTwoArmedDeterministicFixed:
    def test_init(self):
        env = BanditTwoArmedDeterministicFixed()
        assert env.p_dist == [1, 0]
        assert env.r_dist == [1, 1]
        assert env.n_bandits == 2


class TestBanditTwoArmedHighLowFixed:
    def test_init(self):
        env = BanditTwoArmedHighLowFixed()
        assert env.p_dist == [0.8, 0.2]
        assert env.r_dist == [1, 1]
        assert env.n_bandits == 2


class TestBanditTwoArmedHighHighFixed:
    def test_init(self):
        env = BanditTwoArmedHighHighFixed()
        assert env.p_dist == [0.8, 0.9]
        assert env.r_dist == [1, 1]
        assert env.n_bandits == 2


class TestBanditTwoArmedLowLowFixed:
    def test_init(self):
        env = BanditTwoArmedLowLowFixed()
        assert env.p_dist == [0.1, 0.2]
        assert env.r_dist == [1, 1]
        assert env.n_bandits == 2


class TestBanditTwoArmedUniform:
    def test_non_repeatability_without_seed(self):
        env1 = BanditTwoArmedUniform()
        env2 = BanditTwoArmedUniform()
        assert not np.allclose(env1.p_dist, env2.p_dist)

    def test_repeatability_with_seed(self):
        env1 = BanditTwoArmedUniform(seed=42)
        env2 = BanditTwoArmedUniform(seed=42)
        assert np.allclose(env1.p_dist, env2.p_dist)
        assert np.allclose(env1.r_dist, env2.r_dist)


class TestBanditTenArmedRandomFixed:
    def test_non_repeatability_without_seed(self):
        env1 = BanditTenArmedRandomFixed()
        env2 = BanditTenArmedRandomFixed()
        assert not np.allclose(env1.p_dist, env2.p_dist)

    def test_repeatability_with_seed(self):
        env1 = BanditTenArmedRandomFixed(seed=42)
        env2 = BanditTenArmedRandomFixed(seed=42)
        assert np.allclose(env1.p_dist, env2.p_dist)
        assert np.allclose(env1.r_dist, env2.r_dist)


class TestBanditTenArmedUniformDistributedReward:
    def test_non_repeatability_without_seed(self):
        env1 = BanditTenArmedUniformDistributedReward()
        env2 = BanditTenArmedUniformDistributedReward()
        assert not np.allclose(env1.r_dist, env2.r_dist)

    def test_repeatability_with_seed(self):
        env1 = BanditTenArmedUniformDistributedReward(seed=42)
        env2 = BanditTenArmedUniformDistributedReward(seed=42)
        assert np.allclose(env1.p_dist, env2.p_dist)
        assert np.allclose(env1.r_dist, env2.r_dist)


class TestBanditTenArmedRandomRandom:
    def test_non_repeatability_without_seed(self):
        env1 = BanditTenArmedRandomRandom()
        env2 = BanditTenArmedRandomRandom()
        assert not np.allclose(env1.p_dist, env2.p_dist)
        assert not np.allclose(env1.r_dist, env2.r_dist)

    def test_repeatability_with_seed(self):
        env1 = BanditTenArmedRandomRandom(seed=42)
        env2 = BanditTenArmedRandomRandom(seed=42)
        assert np.allclose(env1.p_dist, env2.p_dist)
        assert np.allclose(env1.r_dist, env2.r_dist)


class TestBanditTenArmedGaussian:
    def test_non_repeatability_without_seed(self):
        env1 = BanditTenArmedGaussian()
        env2 = BanditTenArmedGaussian()
        r_dist1 = [r[0] for r in env1.r_dist]
        r_dist2 = [r[0] for r in env2.r_dist]
        assert not np.allclose(r_dist1, r_dist2)

    def test_repeatability_with_seed(self):
        env1 = BanditTenArmedGaussian(seed=42)
        env2 = BanditTenArmedGaussian(seed=42)
        assert np.allclose(env1.p_dist, env2.p_dist)
        r_dist1 = [r[0] for r in env1.r_dist]
        r_dist2 = [r[0] for r in env2.r_dist]
        assert np.allclose(r_dist1, r_dist2)