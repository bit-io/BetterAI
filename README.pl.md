# BetterAI

**Biblioteka uczenia maszynowego dla H# inspirowana PyTorchem** — tensory z broadcastingiem, automatyczne
różniczkowanie (reverse-mode autograd), warstwy sieci neuronowych, optymalizatory, harmonogramy learning rate
i ładowanie danych. Czysty H#, bez kodu natywnego.

```
bit install betterai       # albo `-> betterai => *` w [dependencies] w Bit.hk
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

## Najważniejsze: H# ma semantykę wartości

Tablice i struktury w H# są kopiowane, więc `self.data[i] = v` w metodzie nie zmienia wartości u wywołującego.
PyTorch opiera się na mutacji (`loss.backward()` wypełnia `w.grad`, `opt.step()` zmienia `w`), dlatego BetterAI
ma podobny kształt API, ale **zwraca nowe wartości**:

| PyTorch | BetterAI |
|---|---|
| `loss.backward()`, `w.grad` | `let grads = autograd::backward(loss, model.parameters())` |
| `opt.step()` | `let r = opt.step(model, grads)` → `model = r.model; opt = r.opt` |
| `opt.zero_grad()` | niepotrzebne — gradienty są zwracane, nigdy akumulowane w miejscu |
| `torch.no_grad()`, `.detach()` | `nn::predict(model, x)`, `tensor::detach(t)` |
| pętla zero_grad/forward/loss/backward/step | `utils::train_step(model, opt, x, y, loss_kind, seed)` |

Domknięcia (closures) nie działają jeszcze w backendzie LLVM H#, więc w API ich nie ma — funkcję straty
wybiera się numerem `loss_kind` (`0` mse, `1` cross-entropy, `2` BCE z logitów, `3` L1, `4` Huber, `5` BCE na
prawdopodobieństwach) albo pisze się trzy linijki pętli ręcznie (`examples/02_linear_regression.h#`).

## Moduły
`tensor` (Tensor, tworzenie, broadcasting, matmul, redukcje, aktywacje) · `autograd` (`backward`) ·
`functional` (aktywacje, funkcje straty) · `nn` (Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, GELU, SiLU,
LayerNorm, Dropout, `sequential`, `mlp`) · `optim` (SGD+momentum/Nesterov, Adam, AdamW, RMSprop, Adagrad,
`step_lr`, `exp_lr`, `cosine_lr`, `warmup_lr`) · `data` (Dataset, mini-batche, shuffle, split) ·
`utils` (accuracy, clip_grad_norm, train_step, zapis/odczyt wag).

Pełny opis API: [README.md](README.md). Przykłady: `examples/`. Testy: `tests/` (58 testów, w tym sprawdzanie każdego
gradientu metodą różnic skończonych) — `bit test`.

## Ograniczenia
* **Szybkość** — to skalarny kod H# na tablicach `f64`, a H# kopiuje tablice przy zapisie; biblioteka nadaje się do nauki,
  prototypów i małych modeli, nie do dużych sieci.
* `matmul` i `transpose` tylko dla 2-D; brak konwolucji.
* Testowane w interpreterze H# (`h# preview`) i we wspólnym type-checkerze H# 0.9. Backendu LLVM nie było pod ręką —
  kod unika closures, ale nie był nim kompilowany.

Licencja: BSD-3.
