<div align="center">

<img src="https://media.luminis-sim.com/media/challenge/posters/robodojo_logo.png"></img>

<h2 align="center">RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies</h2>

<h2 align="center"><a href="https://robodojo-benchmark.com/">Webpage</a> | <a href="https://robodojo-benchmark.com/doc/">Document</a> | <a href="https://arxiv.org/abs/2607.04434">Paper</a> | <a href="https://robodojo-benchmark.com/community">Community</a> | <a href="https://robodojo-benchmark.com/leaderboard">Leaderboard</a></h2>

</div>

https://private-user-images.githubusercontent.com/88101805/619409345-cc074c5d-4567-4418-8a29-1385aaba9d5b.mp4

## ✨ Highlights

<p align="center">
  <img src="https://media.luminis-sim.com/media/home/teaser.png" width="70%"></img>
</p>

<p align="center"><em>Overview of RoboDojo. RoboDojo unifies efficient simulation evaluation and reproducible real-world testing for generalist robot manipulation, covering 42 simulation tasks, 18 real-world tasks, heterogeneous parallel simulation, RoboDojo-RealEval, XPolicyLab, and a continuously updated leaderboard.</em></p>

> RoboDojo is **eval-only** in this release: it provides the simulator client, benchmark tasks, asset/config validation, and result artifacts. Policy integration and policy servers are owned by [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md).

- 🌐 **Unified sim-and-real benchmark** — 42 simulation tasks and 18 real-world tasks across 3 robot embodiments for generalist robot manipulation.
- 🧭 **Five capability dimensions** — Generalization, Memory, Precision, Long-Horizon, and Open, designed to probe different skills rather than simple object or layout reskins.
- 🧗 **Challenging by design** — intentionally hard, diverse, long-horizon tasks that expose failures hidden by simpler benchmarks.
- ⚡ **Heterogeneous parallel simulation** — runs different tasks, scenes, and processes concurrently on Isaac Sim for fast, scalable feedback.
- 🧱 **Physically grounded assets** — rigid, articulated, and deformable objects in a single configuration-driven scene.
- 🤖 **Integrate once, evaluate everywhere** — [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md) unifies 40+ policies behind one interface for both simulation and real-world runs.
- 📊 **Reproducible & leaderboard-ready** — seed-controlled layouts and one-command `summarize` aggregation into a leaderboard table.

## 📚 Documentation

The [RoboDojo documentation](https://robodojo-benchmark.com/doc/) is the canonical reference. Key sections:

| Section | Description |
| :-- | :-- |
| [Usage Overview](https://robodojo-benchmark.com/doc/usage/) | End-to-end walkthrough of the evaluation workflow. |
| [Installation & Downloading (Assets and Data)](https://robodojo-benchmark.com/doc/usage/install-and-download/) | Environment setup and downloading robot/object/layout assets/training data. |
| [Quick Evaluation](https://robodojo-benchmark.com/doc/usage/quick-evaluation/) | Quickly dispatch XPolicyLab to run a policy for testing. |
| [XPolicyLab](https://robodojo-benchmark.com/doc/usage/xpolicylab/) | Integrates a large collection of policies and defines how to integrate new ones. |
| [Simulation Tasks Details](https://robodojo-benchmark.com/doc/sim-tasks/) | The 42 Isaac Sim tasks across five capability dimensions. |
| [Real Robot Tasks Details](https://robodojo-benchmark.com/doc/real-tasks/) | The 18 real-world tasks on Piper X, Piper, and ARX X5. |
| [Configurations](https://robodojo-benchmark.com/doc/usage/configurations/) | Simulator, scene, robot, and camera configuration options. |
| [Common Issues](https://robodojo-benchmark.com/doc/common-issue/) | Troubleshooting for installation, assets, GPU memory, and evaluation. |

## 🗂️ Repository Structure

```text
env/                   simulator backbone and managers
env_cfg/               simulator, scene, robot, and camera configs
task/RoboDojo/         task logic and task YAML configs
scripts/robodojo.sh    public RoboDojo-side eval entry
scripts/eval_policy.sh simulator client launched by XPolicyLab eval.sh
XPolicyLab/            policy server and policy integrations
Assets/                downloaded robot, object, material, and layout assets
```

## 🔌 Policy Integration

Policies live in [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md), which owns policy structure, dependencies, checkpoint layout, and server behavior. RoboDojo only assumes a policy directory provides:

```text
XPolicyLab/policy/<POLICY_NAME>/eval.sh
XPolicyLab/policy/<POLICY_NAME>/deploy.yml
```

`eval.sh` starts the policy server and calls back into RoboDojo through `scripts/eval_policy.sh`; `deploy.yml` declares the server host, port, action mode, and policy-specific runtime settings.

## 🏆 Leaderboard

View live rankings on the [RoboDojo Leaderboard](https://robodojo-benchmark.com/leaderboard).

**Simulation.** The full evaluation stack is open source, so you can debug locally and iterate on scores. Official RoboDojo-endorsed listings are submitted through the cloud evaluation pipeline with anti-cheating verification.

**Real world.** Real-robot leaderboard entries are accepted through the same cloud evaluation process; see the public documentation for protocol, rules, and submission details.

## 📝 Citation

**RoboDojo**

```bibtex
@article{chen2026robodojo,
  title={{RoboDojo}: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies},
  author={Chen, Tianxing and Chen, Yue and Li, Zixuan and Tang, Junyuan and Su, Kailun and Wan, Weijie and Chen, Baijun and Lu, Haoran and Yan, Haowen and Su, Honghao and others},
  journal={arXiv preprint arXiv:2607.04434},
  year={2026}
}
```

**RoboTwin 2.0**

```bibtex
@article{chen2025robotwin,
  title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

**MagicSim**

```bibtex
@misc{lu2026magicsimunifiedinfrastructureexecutable,
      title={MagicSim: A Unified Infrastructure for Executable Embodied Interaction}, 
      author={Haoran Lu and Songling Liu and Yue Chen and Guo Ye and Mutian Shen and Shuyang Yu and Yu Xiao and Jihai Zhao and Shang Wu and Jianshu Zhang and Xiangtian Gui and Chuye Hong and Yuran Wang and Maojiang Su and Jiayi Wang and Ruihai Wu and Zhaoran Wang and Han Liu},
      year={2026},
      eprint={2606.17511},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.17511}, 
}
```

## 🙏 Acknowledgements

RoboDojo builds on [Isaac Sim](https://developer.nvidia.com/isaac/sim), [IsaacLab](https://github.com/isaac-sim/IsaacLab), [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena), [RoboTwin 2.0](https://github.com/robotwin-Platform/robotwin), [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), and [MagicSim](https://arxiv.org/abs/2606.17511). We thank the authors and maintainers for their open-source contributions to the robotics community.

Contact [Tianxing Chen](https://tianxingchen.github.io/) or [Yue Chen](https://yuechen0614.github.io/) if you have questions or suggestions.

## 🏫 Affiliations

RoboDojo is operated by **AI MMLab Club**, a non-profit, vendor-neutral organization, and is jointly maintained and supported by a global consortium of academic institutional partners. To preserve the fairness, neutrality, and independence of the official evaluation, RoboDojo does not involve commercial companies in its governance, operation, funding, sponsorship, compute, hardware, or other forms of project support. For inquiries from academic or non-profit partners regarding project collaboration or resource support, please contact [RoboDojoCommittee@gmail.com](mailto:RoboDojoCommittee@gmail.com).

<img src="https://media.luminis-sim.com/media/home/partners/affiliations.png"></img>

## ⚖️ License

Released under the [RoboDojo Non-Commercial Research License](LICENSE). RoboDojo is available for non-commercial research, education, and evaluation only. Commercial use requires prior written permission from the maintainers.


<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-475569?style=flat-square&logo=python&logoColor=white&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Sim 5.1" src="https://img.shields.io/badge/Isaac_Sim-5.1-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Lab 2.3" src="https://img.shields.io/badge/Isaac_Lab-2.3-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="License Non-Commercial" src="https://img.shields.io/badge/License-Non--Commercial-475569?style=flat-square&labelColor=64748b" height="22"/>
</p>
