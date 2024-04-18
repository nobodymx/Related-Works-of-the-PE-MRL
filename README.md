# Added Related Works

**1. Mean Field Game**

Mean field game (MFG) theory studies decision-making for a large population of homogeneous agents with small interactions. The basic idea of the  MFG is to model the game of all agents as a game of two players intertwined with each other, including a representative agent and the empirical distribution of all other agents. The MFG algorithms usually look for the optimal policy in terms of social welfare under cooperative settings [1,2] or Nash Equilibrium policies under competitive/general-sum setting [3]. For example, [1] designs a pessimistic value iteration method under the offline setting to maximize social welfare, and [3] proposes a Q-learning-based algorithm with better convergence stabilities and learning accuracies. However, in this paper, we study the auto-bidding problem **from the perspective of a single advertiser (representative advertiser)** where other advertisers' (background advertisers) policies are assumed to be fixed and known. (Note that this assumption is **viable in practice** since the auto-bidding policy designer, usually the commercial department of an online advertising platform, can access the bidding policy of all advertisers, and there is **no ethical issue** since the observation of the considered advertiser's policy involves no information on other advertisers.) 

**2. Offline Reinforcement Learning**

Offline RL (also known as batch RL) aims to learn better policies based on a fixed offline dataset collected by some behavior policies. The main challenge offline RL addressed is the extrapolation error (also known as the out-of-distribution, OOD) caused by missing data. The key technique to address the extrapolation error is to be pessimistic about the state-action pairs outside the offline dataset and conservatively train the policy. Similar to the traditional online RL algorithms, the offline RL algorithms can be divided into the model-free offline RL and the model-based offline RL.

**Model-free Offline RL** directly trains the policy with the offline dataset without building or learning an environment model. Based on the specific ways of addressing the extrapolation error, the model-free offline RL algorithms can be further divided into policy constraint methods such as BCQ [5], BEAR [6], and conservative regularization methods such as CQL [7], as well as the constrained stationary distribution methods such as AlgaeDICE [8] and OptiDice [9]. Note that the CQL acts as a strong baseline and has been applied to real-world auto-bidding systems [10]. 

**Model-based Offline RL**, in contrast, first learns an environment model and then trains the policy with it. Note that model-free offline RL algorithms can **only learn on the states in the offline dataset**, leading to overly conservative algorithms. Nonetheless, the model-based offline RL algorithms have the potential for **broader generalization** by generating and training on additional imaginary data [11]. The model-based offline algorithms can be divided into two categories. The first category is to build a pessimistic environment model based on the uncertainty estimations, such as MOPO [12], and MOReL [13]. The second one is to apply pessimistic learning methods, e.g., conservative Q learning methods, when training with the learned environment model, such as the COMBO [11] and the H2O [14]. A key factor of model-based offline RL algorithms is the **generalization ability** of the learned environment model since the quality of the generated imaginary data largely determines the policy training. **In this paper, we build a PE environment model, taking advantage of the generalization ability of the PE neural networks.**

**3. Permutation Equivariant/Invariant in RL**

A permutation equivariant (PE) function is a type of function that maintains the same output structure under the reordering of its inputs. Given a set of inputs, if we apply a permutation (rearrange the order of the inputs), a permutation equivariant function will produce a set of outputs that are permuted in the same way as the inputs.
A permutation invariant (PI) function is a type of function that produces the same output regardless of the order of its inputs. In other words, the output of the function is unchanged under any permutation of the input elements.
The PE/PI functions can be implemented by many structures, such as the DeepSet [15], and a set of transformers [18], and have been used in many areas, such as RL, computer visions, and natrual language processings [19,20]. However, most of the previous works employ PE/PI neural networks in modeling policies or value functions in RL algorithms [16,17,21]. For example, [16] uses the Deep Set to model the policies of the UAVs, taking advantage of the variable input dimensions and PI property of the Deep Set, and the policies are trained in a simulated environment implemented by the OpenAI Gym interface that can be freely interacted with. [17] leverages the Deep Set and attention-based neural networks to encode the neighborhood information in the quadrotors' policies and value functions, and the algorithm is trained in a simulated environment that can be freely interacted with. In this paper, we utilize them to model the environment for the agent in RL algorithms, additionally taking advantage of their generalization abilities. Besides, we also highlight that 

* **We are the first to prove the permutation equivariant/invariant (PE/PI) properties of the advertiser's transition rule and the reward function under a typical industrial auction mechanism that involves multiple stages (Proposition 5.1), which provides the basis for the PE environment model design.**
* **We are the first to both theoretically (E.q. 8) and empirically (Table 2 and Table 3) demonstrate the generalization superiority of PE/PI neural networks in modeling the environment in the auto-bidding field.**


**References:**

[1] Chen, M., Li, Y., Wang, E., Yang, Z., Wang, Z., & Zhao, T. (2021). Pessimism meets invariance: Provably efficient offline mean-field multi-agent RL. _Advances in Neural Information Processing Systems_, _34_, 17913-17926.

[2] Li, Y., Wang, L., Yang, J., Wang, E., Wang, Z., Zhao, T., & Zha, H. (2021). Permutation invariant policy optimization for mean-field multi-agent reinforcement learning: A principled approach. _arXiv preprint arXiv:2105.08268_.

[3] Guo, X., Hu, A., Xu, R., & Zhang, J. (2019). Learning mean-field games. _Advances in neural information processing systems_, _32_.

[4] Lasry, J. M., & Lions, P. L. (2007). Mean field games. _Japanese journal of mathematics_, _2_(1), 229-260.

[5] S. Fujimoto, D. Meger, and D. Precup, Off-policy deep reinforcement learning without exploration, in Proceedings of 36th International Conference on Machine Learning, May. 2019.

[6] A. Kumar, J. Fu, M. Soh, G. Tucker, and S. Levine, Stabilizing off-policy q-learning via bootstrapping error reduction. In Advances in Neural Information Processing Systems 32, 2019.

[7] A. Kumar, A. Zhou, G. Tucker, and S. Levine, Conservative q-learning for offline reinforcement learning, in Advances in Neural Information Processing Systems 33, 2020.

[8] Nachum, O., Dai, B., Kostrikov, I., Chow, Y., Li, L., & Schuurmans, D. (2019). Algaedice: Policy gradient from arbitrary experience. _arXiv preprint arXiv:1912.02074_.

[9] Lee, J., Jeon, W., Lee, B., Pineau, J., & Kim, K. E. (2021, July). Optidice: Offline policy optimization via stationary distribution correction estimation. In _International Conference on Machine Learning_ (pp. 6120-6130). PMLR.

[10] Mou, Z., Huo, Y., Bai, R., Xie, M., Yu, C., Xu, J., & Zheng, B. (2022). Sustainable online reinforcement learning for auto-bidding. _Advances in Neural Information Processing Systems_, _35_, 2651-2663.

[11] Yu, T., Kumar, A., Rafailov, R., Rajeswaran, A., Levine, S., & Finn, C. (2021). Combo: Conservative offline model-based policy optimization. _Advances in neural information processing systems_, _34_, 28954-28967.

[12] Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J. Y., Levine, S., ... & Ma, T. (2020). Mopo: Model-based offline policy optimization. _Advances in Neural Information Processing Systems_, _33_, 14129-14142.

[13] Kidambi, R., Rajeswaran, A., Netrapalli, P., & Joachims, T. (2020). Morel: Model-based offline reinforcement learning. _Advances in neural information processing systems_, _33_, 21810-21823.

[14] Niu, H., Qiu, Y., Li, M., Zhou, G., Hu, J., & Zhan, X. (2022). When to trust your simulator: Dynamics-aware hybrid offline-and-online reinforcement learning. _Advances in Neural Information Processing Systems_, _35_, 36599-36612.

[15] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J. (2017). Deep sets. _Advances in neural information processing systems_, _30_.

[16] Logiewa, R., Hoffmann, F., Govaers, F., & Koch, W. (2023, November). Dynamic Pursuit-Evasion Scenarios With a Varying Number of Pursuers Using Deep Sets. In 2023 IEEE Symposium Sensor Data Fusion and International Conference on Multisensor Fusion and Integration (SDF-MFI) (pp. 1-7). IEEE.

[17] Batra, S., Huang, Z., Petrenko, A., Kumar, T., Molchanov, A., & Sukhatme, G. S. (2022, January). Decentralized control of quadrotor swarms with end-to-end deep reinforcement learning. In Conference on Robot Learning (pp. 576-586). PMLR.

[18] Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2018). Set transformer.

[19] Guttenberg, N., Virgo, N., Witkowski, O., Aoki, H., & Kanai, R. (2016). Permutation-equivariant neural networks applied to dynamics prediction. arXiv preprint arXiv:1612.04530.

[20] Gordon, J., Lopez-Paz, D., Baroni, M., & Bouchacourt, D. (2019, September). Permutation equivariant models for compositional generalization in language. In International Conference on Learning Representations.

[21] Tang, Y., & Ha, D. (2021). The sensory neuron as a transformer: Permutation-invariant neural networks for reinforcement learning. Advances in Neural Information Processing Systems, 34, 22574-22587.
