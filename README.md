# Delta-AI
Code for our paper [Delta-AI: Local objectives for amortized inference in sparse graphical models](https://arxiv.org/abs/2310.02423) 
by [Jean-Pierre Falet](https://www.jeanpierrefalet.com), [Hae Beom Lee](https://haebeom-lee.github.io), [Nikolay Malkin](https://malkin1729.github.io/), [Chen Sun](https://linclab.mila.quebec/team/chen), Dragos Secrieru, [Dinghuai Zhang](https://zdhnarsil.github.io/), [Guillaume Lajoie](https://www.guillaumelajoie.com), and [Yoshua Bengio](https://yoshuabengio.org/).


### Examples

Synthetic tasks

```
python synthetic/main_delta.py --alg rand --model ising --graph lattice --vdim 264 --epsilon 1 --temp 10 --glr 1e-3 --mlr 1e-1
```

Latent-variable modeling (MNIST)

```
python mnist/main_delta.py --alg rand --sampling_dag partial --epsilon 0.05 --temp 4 --q_lr 1e-3 --p_lr 1e-3 --marg_q_lr 1e-1 --marg_p_lr 1e-1 --q_objective delta
```
