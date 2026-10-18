# BetterAI

A **PyTorch-inspired machine-learning library for H#** — tensors with broadcasting, reverse-mode
automatic differentiation, neural-network layers, optimizers, learning-rate schedules and data
loading. Pure H#, no native code. Polish version: [README.pl.md](README.pl.md).

```
bit install betterai            # or add `-> betterai => *` to [dependencies] in Bit.hk
```

```hsharp
use "bit -> betterai"

fn main() is
    let x: tensor::Tensor = tensor::from_rows([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    let y: tensor::Tensor = tensor::from_rows([[0.0], [1.0], [1.0], [0.0]])

    let mut model: nn::Module = nn::sequential([nn::linear(2, 8, 1), nn::tanh(), nn::linear(8, 1, 2)])
    let mut opt: optim::Optimizer = optim::adam(0.05)

    let mut epoch: int = 0
    while epoch < 300 is
        let r: utils::TrainStep = utils::train_step(model, opt, x, y, 2, epoch)
        model = r.model
        opt = r.opt
        epoch += 1
    end
    write(nn::predict(model, x).sigmoid().show())
end
```

## The one thing to know: H# has value semantics

In H# arrays and structs are copied, and `self.data[i] = v` inside a method does not change the
caller's value. PyTorch relies on mutation (`loss.backward()` fills `w.grad`, `opt.step()` edits `w`),
so BetterAI keeps the same *shape* of API but returns new values:

| PyTorch | BetterAI |
|---|---|
| `loss.backward()` then `w.grad` | `let grads = autograd::backward(loss, model.parameters())` |
| `opt.step()` | `let r = opt.step(model, grads)`  then  `model = r.model; opt = r.opt` |
| `opt.zero_grad()` | not needed — gradients are returned, never accumulated in place |
| `torch.no_grad()` / `.detach()` | `nn::predict(model, x)`, `tensor::detach(t)` — constants never record a graph |
| `model.train()` / `.eval()` | `nn::train(model)` / `nn::eval(model)` |
| loop of zero_grad / forward / loss / backward / step | `utils::train_step(model, opt, x, y, loss_kind, seed)` |

Closures and function values are not available in H#'s LLVM backend yet, so there are none in this
API: losses are chosen with a `loss_kind` number (see below) or you write the three lines of the loop
yourself, as in `examples/02_linear_regression.h#`.

## Modules

They are inlined into your program and used by plain name: `tensor::`, `autograd::`, `functional::`,
`nn::`, `optim::`, `data::`, `utils::`.

### `tensor` — `tensor::Tensor`
Immutable n-d array of `f64`, row-major, `shape: [int]`.

* creation: `from_vec`, `vector`, `from_rows`, `scalar`, `zeros`, `ones`, `full`, `zeros_like`,
  `ones_like`, `arange`, `linspace`, `eye`, `rand`, `uniform`, `randn` (all random functions take a `seed`)
* parameters: `param(data, shape, leaf_id)`, `with_leaf`, `detach`
* math with numpy-style **broadcasting**: `add sub mul div neg mul_scalar add_scalar pow sqrt exp log abs clamp`
* matrices: `matmul` (2-D), `transpose`, `reshape` (with `-1`), `flatten`
* reductions: `sum mean sum_axis mean_axis argmax_rows`
* activations: `relu leaky_relu sigmoid tanh softmax log_softmax`, `nll`
* inspection: `item shape numel get get2 show print`
* methods, so code reads like PyTorch: `x.matmul(w).add(b).relu()`, `.t() .sum() .mean() .softmax() .scale(c) .shift(c) ...`

### `autograd`
`autograd::backward(loss, params) -> [Tensor]` returns d(loss)/d(param) in the order of `params`
(zeros for parameters the loss does not depend on). `loss` must be a scalar. Parameters are the tensors
carrying a `leaf` id (`tensor::param`, or everything inside a `nn::Module`).

### `functional`
`relu leaky_relu sigmoid tanh softmax log_softmax softplus gelu silu dropout one_hot normalize` and the losses
`mse_loss l1_loss huber_loss bce_loss bce_with_logits nll_loss cross_entropy`.

### `nn`
Layers: `linear` (PyTorch-style init), `linear_xavier`, `linear_kaiming`, `relu sigmoid tanh leaky_relu softmax gelu silu softplus flatten`,
`layer_norm(features)`, `dropout(p, seed)`.
`nn::sequential(layers)` builds a `Module` and numbers its parameters; `nn::mlp([in, h1, ..., out], act, seed)`
builds a fully connected net. Use `sequential_at(layers, base)` when two models live in one graph (GAN, encoder/decoder)
so their parameter ids do not collide.
`model.forward(x)`, `forward_seeded(x, seed)` (vary dropout masks), `parameters()`, `param_count()`, `with_parameters(ps)`,
`train()`, `eval()`, `summary()`, `nn::predict(model, x)` (eval + no graph).
Linear weights are `[in, out]`: `y = x · W + b`.

### `optim`
`sgd(lr, momentum)`, `sgd_nesterov`, `adam`, `adamw(lr, weight_decay)`, `rmsprop(lr, alpha)`, `adagrad`;
`opt.step(model, grads)` / `step_params(opt, params, grads)` return the new parameters **and** the updated optimizer
(momentum / Adam moments). Schedules are pure functions you feed into `opt.with_lr(...)`:
`step_lr`, `exp_lr`, `cosine_lr`, `warmup_lr`.

### `data`
`dataset(x, y)`, `batches(ds, batch_size, seed)` (a negative seed keeps the order), `split(ds, train_fraction, seed)`,
`permutation`, `select_rows`, `labels(y)`.

### `utils`
`accuracy`, `binary_accuracy`, `r2_score`, `grad_norm`, `clip_grad_norm`, `train_step`, `evaluate`,
`save_params(path, model)` / `load_params(path, model)` (plain-text weights).
`loss_kind` for `train_step`/`evaluate`: `0` mse · `1` cross-entropy (targets are class ids stored as floats) ·
`2` BCE with logits · `3` L1 · `4` Huber · `5` BCE on probabilities.

## Examples (`examples/`)
| file | shows |
|---|---|
| `01_autograd.h#` | gradients of a hand-built graph |
| `02_linear_regression.h#` | parameters + loss + backward + optimizer written out by hand |
| `03_xor.h#` | `nn::sequential` + `train_step` |
| `04_classifier.h#` | train/test split, shuffled mini-batches, cosine LR, gradient clipping, saving weights |

Run one with `h# preview examples/03_xor.h#` (the interpreter) after the library is installed, or build it with `bit`.

## Tests
`tests/*.h#` hold 58 `#[test]` functions: every differentiable op is checked against central finite
differences, plus broadcasting, layers, losses, all optimizers (convergence and exact single steps),
data loading, schedules, clipping and weight (de)serialisation. Run them with `bit test`
(`[dev-dependencies]` links this directory as `betterai`).

## Limits worth knowing
* **Speed.** Everything is scalar H# code over `f64` arrays and H# copies arrays on write, so this is for learning,
  prototyping and small models (hundreds to a few thousand parameters), not for large networks. XOR trains in
  seconds under `h# preview`.
* **Backward pass cost.** The graph is walked recursively; each path to a parameter is visited once. Layer chains
  (the normal case) visit each node once. A hand-written expression that fans one intermediate value out into many
  branches is walked once per branch.
* `matmul`/`transpose` are 2-D only; use `reshape` for vectors. No convolutions yet.
* Tested with the H# interpreter (`h# preview`) and the shared type-checker of H# 0.9. The LLVM backend was not available
  when this was written; the code avoids closures (not implemented there) but has not been compiled with it.

## Layout
```
Bit.hk        package manifest
src/lib.h#    entry point (declares the modules)
src/tensor.h#  autograd.h#  functional.h#  nn.h#  optim.h#  data.h#  utils.h#
examples/     tests/     index-entry.json (a ready entry for bit.io's index/repository.json)
```

License: MIT.
