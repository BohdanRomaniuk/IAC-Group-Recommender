# IAC-Group-Recommender

**Інтегратор–Актор–Критик (Integrator–Actor–Critic, IAC): система групових рекомендацій на основі глибокого навчання з підкріпленням**

> *Науково-дослідний проєкт. Версія 2.0 · Травень 2026*

---

## Анотація

У цій роботі запропоновано архітектуру **Інтегратор–Актор–Критик (Integrator–Actor–Critic, IAC)** — систему групових рекомендацій кінофільмів, що поєднує методи глибокого навчання з підкріпленням (Deep Reinforcement Learning, DRL) із механізмом уваги (attention) для агрегації вподобань членів групи. Агент навчається на наборі даних MovieLens, взаємодіючи з середовищем, побудованим на основі невід'ємного матричного розкладу (Non-negative Matrix Factorization, NMF). Використання пріоритизованого буфера досвіду (Prioritized Experience Replay, PER), ентропійної регуляризації (entropy regularization) актора та косинусного відпалу (cosine annealing) разом з оптимізатором Ranger дозволяє досягти стабільного та ефективного навчання.

---

## Зміст

1. [Мотивація та постановка задачі](#1-мотивація-та-постановка-задачі)
2. [Архітектура системи](#2-архітектура-системи)
3. [Методологія](#3-методологія)
4. [Структура проєкту](#4-структура-проєкту)
5. [Встановлення та залежності](#5-встановлення-та-залежності)
6. [Встановлення середовища та запуск проєкту](#6-встановлення-середовища-та-запуск-проєкту)
7. [Конфігурація гіперпараметрів](#7-конфігурація-гіперпараметрів)
8. [Метрики оцінювання](#8-метрики-оцінювання)
9. [Результати та артефакти](#9-результати-та-артефакти)
10. [Препроцесинг даних](#10-препроцесинг-даних)

---

## 1. Мотивація та постановка задачі

Традиційні системи рекомендацій (recommender systems) орієнтовані на індивідуального користувача і не враховують колективних уподобань групи людей (наприклад, вибір фільму для перегляду компанією). Задача **групової рекомендації** (group recommendation) полягає у формуванні ранжованого списку елементів, що максимізують колективне задоволення членів групи за наявності конфліктних уподобань.

Формально задача визначається як **марковський процес прийняття рішень** (Markov Decision Process, MDP):

- **Стан** (state) `s_t` — ідентифікатор поточної групи разом з її зваженою за увагою агрегованою репрезентацією вподобань та ембеддингами (embeddings) останніх `H` взаємодій.
- **Дія** (action) `a_t` — неперервний вектор у просторі ембеддингів, найближчий сусід (nearest neighbor) до якого визначає рекомендований елемент.
- **Винагорода** (reward) `r_t` — бінарний сигнал релевантності (0 або 1), отриманий з NMF-реконструкції матриці оцінок, доповнений штрафами за різноманітність (diversity) та покриття каталогу (catalogue coverage).
- **Мета** — максимізація довгострокової накопиченої винагороди (cumulative return) `G_t = Σ γ^k · r_{t+k}` з коефіцієнтом дисконтування (discount factor) `γ = 0.95`.

---

## 2. Архітектура системи

### 2.1 Загальна схема

```
         ┌──────────────────────────────────────────────────────┐
         │                Інтегратор (Integrator)               │
         │  ┌─────────────┐   Multi-Head    ┌───────────────┐  │
Члени    │  │  Ембеддинги │   Self-Attention│   Гейтоване   │  │
групи ──►│  │ користувачів│───────────────►│  агрегування  │──►  Стан s_t
         │  │  (E × U)    │                │ (Gated Resid.)│  │
         │  │  Ембеддинги │                └───────────────┘  │
         │  │   фільмів   │                                    │
         │  │  (E × I)    │                                    │
         │  └─────────────┘                                    │
         └──────────────────────────────────────────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                                       ▼
         ┌──────────┐                           ┌──────────┐
         │  Актор   │  ──── a_t ────────────►  │  Критик  │
         │ (Actor)  │                           │ (Critic) │
         │  π(s_t)  │                           │  Q(s,a)  │
         └──────────┘                           └──────────┘
               │                                       │
               └─── TD-похибка ──► Буфер PER ◄─────────┘
```

### 2.2 Компоненти

| Компонент | Клас | Призначення |
|---|---|---|
| **Інтегратор** (Integrator) | `models/integrator.py` | Таблиці ембеддингів користувачів/елементів + увага для агрегації групи |
| **Мультиголовне групове агрегування** (MultiHeadGroupAggregation) | `models/integrator.py` | Мультиголовна самоувага (multi-head self-attention) з гейтованим залишковим з'єднанням (gated residual connection) |
| **Актор** (Actor) | `models/actor.py` | Відображення стану у простір дій: `π: s → a` |
| **Критик** (Critic) | `models/critic.py` | Оцінка функції цінності дії (Q-function): `Q: (s, a) → ℝ` |
| **Агент DDPG** (DDPGRecommenderAgent) | `agent/ddpg.py` | Оркестрація навчання: DDPG + PER + ентропійна регуляризація |
| **Пріоритизований буфер досвіду** (PrioritizedExperienceReplay) | `agent/replay_buffer.py` | Буфер досвіду з пропорційним пріоритетом за TD-похибкою |
| **Середовище рекомендацій** (RecommendationEnvironment) | `environment/env.py` | Gymnasium-середовище з NMF-реконструйованою матрицею винагород |
| **Завантажувач даних** (MovieLensDatasetLoader) | `environment/dataloader.py` | Завантаження та обробка даних MovieLens (групи / оцінки / негативні зразки) |
| **Тренер** (Trainer) | `environment/trainer.py` | Повний цикл навчання з контрольними точками (checkpointing) |
| **Оцінювач Top-K метрик** (TopKMetricsEvaluator) | `environment/evaluator.py` | Обчислення Recall@K, NDCG@K, Precision@K, MAP, HR, MRR, Coverage |
| **Шум Орнштейна–Уленбека** (OrnsteinUhlenbeckNoise) | `utils/noise.py` | Корельований у часі шум для дослідження простору дій |
| **Візуалізація втрат** (`plot_losses`) | `visualization/plot.py` | Побудова графіків функцій втрат актора та критика |

---

## 3. Методологія

### 3.1 Агрегація групових уподобань

Для агрегації різнорідних уподобань членів групи застосовано **мультиголовну самоувагу** (Multi-Head Self-Attention, MHSA) із гейтованим залишковим з'єднанням (gated residual connection). На відміну від тривіального усереднення (mean pooling), цей механізм дозволяє членам групи «звертати увагу» один на одного, що моделює соціальну динаміку прийняття спільних рішень:

```
Attended  =  MHSA(X, X, X)
Gate      =  σ(Linear([Attended ∥ X]))
Output    =  Gate ⊙ Attended + (1 − Gate) ⊙ X
Group_vec =  mean(Output)
```

де `X ∈ ℝ^{|G| × E}` — матриця ембеддингів членів групи `G`, `E` — розмірність ембеддингу.

### 3.2 Алгоритм DDPG з розширеннями

Навчання агента базується на алгоритмі **глибокого детермінованого градієнта політики** (Deep Deterministic Policy Gradient, DDPG) із такими розширеннями:

1. **Пріоритизований буфер досвіду** (Prioritized Experience Replay, PER) — переходи з більшою TD-похибкою (temporal-difference error) δ = |r + γQ'(s', π'(s')) − Q(s, a)| відбираються частіше; для корекції зміщення вибірки використовуються ваги важливісної вибірки (importance sampling, IS) з анілюванням коефіцієнта `β` від 0.4 до 1.0.

2. **Ентропійна регуляризація актора** (entropy regularization) — до функції втрат (loss function) актора додається ентропійний бонус `−α_ent · H(π)`, що стимулює дослідження (exploration) та запобігає передчасній збіжності (premature convergence).

3. **Формування винагороди з різноманітністю та покриттям** (reward shaping) — винагорода доповнюється:
   ```
   r̃ = r + α_div · Diversity(a_t, history) + β_cov · Coverage(recommended)
   ```

4. **Оптимізатор Ranger + косинусний відпал** (Ranger optimizer + cosine annealing) — поєднання RAdam із Lookahead (Ranger) та планувальника швидкості навчання `CosineAnnealingLR` для плавної та стійкої оптимізації ваг.

5. **Цільові мережі з м'яким оновленням** (target networks with soft update) — параметри цільових мереж оновлюються за правилом Поляка (Polyak averaging): `θ' ← τ·θ + (1 − τ)·θ'`, де `τ = 0.005`.

6. **Шум Орнштейна–Уленбека** (Ornstein–Uhlenbeck process, OU) — корельований у часі дослідницький шум на просторі дій для кращого дослідження в середовищах з неперервними діями.

### 3.3 Середовище на основі NMF

Для апроксимації істинної матриці вподобань використовується **невід'ємний матричний розклад** (Non-negative Matrix Factorization, NMF) із рангом, що дорівнює `embedding_size`. Отримана матриця винагород кешується у файл `data/saves/env_<split>_<dim>.npy` для прискорення подальших запусків.

---

## 4. Структура проєкту

```
IAC-Group-Recommender/
│
├── main.py                     # Точка входу: збирає компоненти та запускає Тренер
├── config.py                   # TrainingConfig — централізований контейнер гіперпараметрів
├── default.yaml                # Базові гіперпараметри (перевизначаються через --config)
│
├── models/                     # Нейронні мережі (neural networks)
│   ├── actor.py                # Актор: багатошаровий перцептрон (MLP) s → a
│   ├── critic.py               # Критик: MLP (s, a) → Q
│   └── integrator.py           # Ембеддинги + MHSA-агрегація групи
│
├── agent/                      # Агент навчання з підкріпленням (RL agent)
│   ├── ddpg.py                 # DDPGRecommenderAgent
│   └── replay_buffer.py        # Пріоритизований буфер досвіду (Schaul et al., 2016)
│
├── environment/                # Середовище, дані, навчання та оцінювання
│   ├── env.py                  # Середовище рекомендацій (Gymnasium + NMF)
│   ├── dataloader.py           # Завантажувач набору даних MovieLens
│   ├── trainer.py              # Тренер — повний цикл навчання
│   └── evaluator.py            # Оцінювач Top-K метрик
│
├── utils/
│   ├── noise.py                # Шум Орнштейна–Уленбека
│   └── helpers.py              # Допоміжний рівномірний буфер (для зворотної сумісності)
│
├── visualization/
│   └── plot.py                 # plot_losses() — криві втрат у форматі PNG
│
├── generators/                 # Скрипти препроцесингу (preprocessing) наборів даних
│   ├── generator.py            # MovieLens-1M
│   └── generator_32m.py        # MovieLens-32M
│
└── data/
    ├── ml-1m/                  # Вихідний (raw) набір даних MovieLens-1M
    ├── ml-32m/                 # Вихідний набір даних MovieLens-32M
    ├── MovieLens-1m/           # Оброблені дані з випадковим розбиттям груп
    ├── MovieLens-32m/          # Оброблені дані MovieLens-32M
    ├── saves/                  # Контрольні точки моделей, кеш eval-таблиць, NMF-середовище
    └── results/                # Криві втрат, числові метрики
```

> **Примітка:** Проєкт використовує **неявні namespace-пакети** (implicit namespace packages, PEP 420) — файли `__init__.py` відсутні. Python 3.3+ підтримує це нативно. Усі команди слід виконувати з кореневої директорії проєкту.

---

## 5. Встановлення та залежності

Ключові залежності: `torch`, `gymnasium`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `pytorch-optimizer`, `pyyaml`, `matplotlib`.

Повний перелік зафіксовано у `requirements.txt`. Детальний процес встановлення та запуску описано у [розділі 6](#6-встановлення-середовища-та-запуск-проєкту).

---

## 6. Встановлення середовища та запуск проєкту

### 6.1 Вимоги до системи

| Компонент | Мінімальна версія |
|---|---|
| Python | 3.10 |
| pip | 23.0 |
| CUDA (опціонально) | 11.8 |
| Оперативна пам'ять (RAM) | 8 ГБ |
| Відеопам'ять (VRAM, опціонально) | 4 ГБ |

### 6.2 Створення та активація віртуального середовища

Рекомендується ізолювати залежності проєкту у віртуальному середовищі (virtual environment):

```bash
# Перейти до кореневої директорії проєкту
cd IAC-Group-Recommender

# Створити віртуальне середовище
python -m venv .venv
```

**Активація:**

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

Після успішної активації запрошення (prompt) терміналу набуде вигляду `(.venv) user@host:~/...`.

**Деактивація** (після завершення роботи):

```bash
deactivate
```

### 6.3 Встановлення залежностей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Для перевірки, що PyTorch коректно бачить графічний процесор (GPU):

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); \
           print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 6.4 Підготовка даних (виконується один раз)

Перед першим запуском необхідно згенерувати оброблені групові набори даних із вихідних (raw) файлів MovieLens:

```bash
# MovieLens-1M  →  data/MovieLens-1m/
python -m generators.generator

# MovieLens-32M  →  data/MovieLens-32m/
python -m generators.generator_32m
```

Після завершення у відповідній директорії мають бути присутні файли:
`groupMember.dat`, `groupRatingTrain.dat`, `groupRatingVal.dat`, `groupRatingTest.dat`,
`groupRatingValNegative.dat`, `groupRatingTestNegative.dat`, `userRatingTrain.dat`,
`userRatingVal.dat`, `userRatingTest.dat`, `userRatingValNegative.dat`,
`userRatingTestNegative.dat`, `movies.dat`, `users.dat`.

### 6.5 Запуск навчання моделі

**З параметрами за замовчуванням** (гіперпараметри беруться безпосередньо з `config.py`):

```bash
python main.py
```

**З YAML-конфігом** (рекомендований спосіб для відтворюваних експериментів — reproducible experiments):

```bash
python main.py --config default.yaml
```

**Запуск власного експерименту з модифікованими параметрами:**

```bash
# Скопіювати базовий конфіг
cp default.yaml my_experiment.yaml

# Відредагувати потрібні поля, наприклад:
#   embedding_size: 128
#   max_episodes: 2000
#   entropy_coef: 0.01

python main.py --config my_experiment.yaml
```

Під час навчання у консоль виводяться структуровані логи (logs):

```
2026-05-01 12:00:00 | INFO    | environment.trainer | Episode  10/1000 | actor_loss: 0.3421 | critic_loss: 0.8103
2026-05-01 12:00:05 | INFO    | environment.evaluator | [group] Recall@10: 0.1523 | NDCG@10: 0.1847 | HR@10: 0.4201
2026-05-01 12:00:05 | INFO    | environment.trainer | ✓ New best — checkpoint saved (Group Recall@10: 0.1523)
```

Найкраща контрольна точка (best checkpoint) за **Group Recall@10** автоматично зберігається у `data/saves/`.

### 6.6 Завершення та збережені артефакти

Після завершення навчання у `data/saves/` та `data/results/` знаходитимуться:

```
data/
├── saves/
│   ├── best_actor.pt          ← ваги актора (найкраща контрольна точка)
│   ├── best_critic.pt         ← ваги критика
│   ├── best_embedding.pt      ← ваги Інтегратора + модуля уваги
│   ├── env_val_64.npy         ← кешована NMF-матриця винагород
│   ├── eval_user_test_10.pkl  ← кешована eval-таблиця (користувачі)
│   └── eval_group_test_10.pkl ← кешована eval-таблиця (групи)
└── results/
    ├── Proposed_64.txt        ← числові метрики за всіма значеннями K
    └── *_loss_curves.png      ← PNG-графіки кривих втрат
```

---

## 7. Конфігурація гіперпараметрів

Усі гіперпараметри (hyperparameters) зосереджені в класі `TrainingConfig` (`config.py`) і можуть бути перевизначені через YAML-файл.

| Параметр | За замовч. | Опис |
|---|---|---|
| `embedding_size` | `64` | Розмірність ембеддингу (embedding dimension) `E` |
| `history_length` | `10` | Довжина вікна взаємодій (interaction history window) `H` |
| `max_episodes` | `1000` | Кількість епізодів (episodes) навчання |
| `steps_per_episode` | `100` | Кроків середовища (environment steps) на епізод |
| `gradient_steps_per_env_step` | `2` | Кроків градієнту (gradient steps) на крок середовища |
| `warmup_steps` | `500` | Кроків випадкового заповнення буфера (warm-up) |
| `batch_size` | `128` | Розмір мінібатчу (mini-batch size) |
| `buffer_size` | `100 000` | Ємність буфера PER (buffer capacity) |
| `tau` | `0.005` | Коефіцієнт м'якого оновлення цільової мережі (soft update rate) |
| `gamma` | `0.95` | Коефіцієнт дисконтування (discount factor) |
| `actor_hidden_sizes` | `(256, 128)` | Розміри прихованих шарів (hidden layers) актора |
| `critic_hidden_sizes` | `(256, 128)` | Розміри прихованих шарів критика |
| `actor_learning_rate` | `1e-3` | Швидкість навчання (learning rate) актора |
| `critic_learning_rate` | `1e-3` | Швидкість навчання критика |
| `embedding_learning_rate` | `5e-4` | Швидкість навчання Інтегратора |
| `entropy_coef` | `0.05` | Коефіцієнт ентропійної регуляризації `α_ent` |
| `reward_diversity_alpha` | `0.5` | Вага бонусу за різноманітність (diversity bonus) |
| `reward_coverage_beta` | `0.1` | Вага бонусу за покриття каталогу (coverage bonus) |
| `per_alpha` | `0.6` | Ступінь пріоритизації у PER (`0` — рівномірна вибірка) |
| `per_beta_start` | `0.4` | Початкове значення `β` для IS-ваг (importance sampling weights) |
| `ou_theta` / `ou_sigma` | `0.15` / `0.3` | Параметри шуму Орнштейна–Уленбека |
| `eval_top_k_values` | `[5,10,15,20]` | Значення `K` для обчислення Top-K метрик |
| `eval_interval_episodes` | `10` | Частота оцінювання (evaluation interval, у епізодах) |

---

## 8. Метрики оцінювання

Оцінювання виконується класом `TopKMetricsEvaluator` для двох режимів: **користувач** (`user`) та **група** (`group`). Для кожного значення `K` обчислюються такі метрики:

| Метрика | Позначення | Опис |
|---|---|---|
| **Повнота** (Recall@K) | `Recall@K` | Частка релевантних елементів, що потрапили до топ-K рекомендацій |
| **Нормалізований дисконтований кумулятивний виграш** (NDCG@K) | `NDCG@K` | Якість ранжування з урахуванням позиції у списку |
| **Точність** (Precision@K) | `Prec@K` | Точність у топ-K рекомендаціях |
| **Середня усереднена точність** (Mean Average Precision, MAP@K) | `MAP@K` | Середня точність по всіх запитах |
| **Коефіцієнт влучення** (Hit Rate@K) | `HR@K` | Частка запитів, де хоча б один релевантний елемент у топ-K |
| **Середній обернений ранг** (Mean Reciprocal Rank, MRR@K) | `MRR@K` | Середнє значення оберненого рангу першого релевантного елемента |
| **Покриття каталогу** (Catalogue Coverage) | `Cov` | Частка унікальних елементів каталогу, рекомендованих протягом тесту |

Контрольна точка зберігається автоматично щоразу, коли **Group Recall@10** перевищує попереднє найкраще значення.

---

## 9. Результати та артефакти

Після завершення навчання генеруються такі артефакти (artifacts):

| Артефакт | Шлях | Опис |
|---|---|---|
| Ваги актора (actor weights) | `data/saves/best_actor.pt` | Найкраща контрольна точка за Group Recall@10 |
| Ваги критика (critic weights) | `data/saves/best_critic.pt` | Відповідна контрольна точка критика |
| Ваги Інтегратора | `data/saves/best_embedding.pt` | Ваги ембеддингів та модуля уваги |
| NMF-середовище | `data/saves/env_<split>_<dim>.npy` | Кешована матриця винагород NMF |
| Eval-таблиці | `data/saves/eval_{user,group}_{val,test}_<H>.pkl` | Кешовані DataFrame-и для оцінювання |
| Криві втрат (loss curves) | `data/results/<timestamp>_loss_curves.png` | PNG-графік втрат актора та критика |
| Числові метрики | `data/results/` | Логи метрик (`Proposed_64.txt` тощо) |

---

## 10. Препроцесинг даних

Скрипти у каталозі `generators/` трансформують вихідні (raw) файли MovieLens у формат, придатний для навчання: генерують групи користувачів, розбивають оцінки на тренувальну (train), валідаційну (validation) і тестову (test) вибірки та формують файли негативних зразків (negative samples) для оцінювання.

```bash
# Генерація для MovieLens-1M
python -m generators.generator

# Генерація для MovieLens-32M
python -m generators.generator_32m
```

Очікувані вихідні файли у відповідній директорії: `groupMember.dat`, `groupRatingTrain.dat`, `groupRatingVal.dat`, `groupRatingTest.dat`, `groupRatingValNegative.dat`, `groupRatingTestNegative.dat`, `userRatingTrain.dat`, `userRatingVal.dat`, `userRatingTest.dat`, `userRatingValNegative.dat`, `userRatingTestNegative.dat`, `movies.dat`, `users.dat`.
