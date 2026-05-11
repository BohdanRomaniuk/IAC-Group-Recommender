# Uploading Group Recommender Data to BigQuery

This guide walks through uploading the exported CSV files to BigQuery and connecting them to Looker Studio for visualization. No prior BigQuery experience required.

## What you will end up with

Six tables in BigQuery inside a dataset called `group_recommender`:

| Table | Size | Description |
|---|---|---|
| `groups` | 1 000 rows | One row per synthetic group with its member count |
| `group_members` | 2 215 rows | One row per (group, user) membership pair |
| `group_ratings` | 954 230 rows | Group–movie interactions (train / val / test splits) |
| `user_ratings` | 9 748 457 rows | Individual user–movie interactions |
| `users` | 200 948 rows | User IDs (demographic fields are dummy data and excluded) |
| `movies` | 154 169 rows | Movie metadata — one row per genre, so each movie appears multiple times |

---

## Step 0 — Prerequisites

- A Google account (Gmail is fine)
- The CSV files produced by the export script, located at:
  ```
  src/generators/data/MovieLens-32m/looker_csvs/
  ```
  If the folder does not exist yet, run from `src/`:
  ```bash
  python -m generators.export_looker_csvs
  ```

---

## Step 1 — Create a Google Cloud project

1. Open [console.cloud.google.com](https://console.cloud.google.com) and sign in.
2. Click the project selector at the top of the page → **New Project**.
3. Give it a name, e.g. `group-recommender`, then click **Create**.
4. Make sure the new project is selected in the top bar before continuing.

> BigQuery is free for the first 10 GB of storage and 1 TB of queries per month. This dataset is well under 1 GB, so no charges should apply.

---

## Step 2 — Enable the BigQuery API

1. In the left sidebar go to **APIs & Services → Library**.
2. Search for **BigQuery API** and click **Enable**.

> On new projects the BigQuery API is often already enabled. If you see a **Manage** button instead of **Enable**, you are good.

---

## Step 3 — Install the Google Cloud CLI

The BigQuery web console can only upload files up to 10 MB directly. Two of the CSV files (`group_ratings.csv` at 25 MB and `user_ratings.csv` at 278 MB) exceed that limit, so you need the command-line tool (`bq`), which ships as part of the Google Cloud CLI.

### Linux

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL          # reload shell so gcloud is on PATH
```

### Verify the install

```bash
gcloud --version
bq version
```

Both commands should print a version number.

---

## Step 4 — Authenticate

```bash
gcloud auth login
```

A browser window opens. Sign in with your Google account and grant access.

Then set your project so that `bq` commands target the right place:

```bash
gcloud config set project YOUR_PROJECT_ID
```

Replace `YOUR_PROJECT_ID` with the ID shown in the Google Cloud console (it looks like `group-recommender-123456`). You can also find it by running:

```bash
gcloud projects list
```

---

## Step 5 — Create a BigQuery dataset

A **dataset** is a container for tables, similar to a database schema.

```bash
bq mk --location=US group_recommender
```

> If you are in Europe and prefer data stored there, replace `US` with `EU`.

Confirm it was created:

```bash
bq ls
```

You should see `group_recommender` in the output.

---

## Step 6 — Upload the CSV files

Set a variable pointing to the CSV folder so the commands below are shorter:

```bash
CSV_DIR="$HOME/Projects/IAC-Group-Recommender/src/generators/data/MovieLens-32m/looker_csvs"
```

Now load each table. The `--autodetect` flag tells BigQuery to infer column names and types from the CSV header automatically.

```bash
bq load --autodetect --source_format=CSV \
    group_recommender.groups \
    "$CSV_DIR/groups.csv"

bq load --autodetect --source_format=CSV \
    group_recommender.group_members \
    "$CSV_DIR/group_members.csv"

bq load --autodetect --source_format=CSV \
    group_recommender.group_ratings \
    "$CSV_DIR/group_ratings.csv"

bq load --autodetect --source_format=CSV \
    group_recommender.user_ratings \
    "$CSV_DIR/user_ratings.csv"

bq load --autodetect --source_format=CSV \
    group_recommender.users \
    "$CSV_DIR/users.csv"

bq load --autodetect --source_format=CSV \
    group_recommender.movies \
    "$CSV_DIR/movies.csv"
```

Each command prints a job ID and then `SUCCESS` when done. The two large files (`group_ratings`, `user_ratings`) may take a minute or two.

### Verify the upload

```bash
bq ls group_recommender
bq show group_recommender.group_ratings
```

The first command lists all tables; the second shows the detected schema for `group_ratings`. You should see columns: `group_id`, `movie_id`, `rating`, `timestamp`, `split`.

You can also run a quick sanity query:

```bash
bq query --use_legacy_sql=false \
  'SELECT split, COUNT(*) AS cnt FROM group_recommender.group_ratings GROUP BY split ORDER BY split'
```

Expected output:

```
+---------------+---------+
|     split     |   cnt   |
+---------------+---------+
| test          |    6096 |
| test_negative |  609600 |
| train         |   21338 |
| val           |    3050 |
| val_negative  |  305000 |
+---------------+---------+
```

---

## Step 7 — Connect Looker Studio

1. Go to [lookerstudio.google.com](https://lookerstudio.google.com).
2. Click **Create → Report**.
3. In the connector list, choose **BigQuery**.
4. Select **My Projects → group-recommender → group_recommender → group_ratings** (or whichever table you want to start with).
5. Click **Add** in the bottom right → **Add to report**.

To add more tables to the same report, click **Add data** in the toolbar and repeat from step 3.

> Looker Studio joins tables on shared fields (e.g. `movie_id`, `group_id`) using **Blended data**. Go to **Resource → Manage blended data** to configure joins between tables.

---

## Useful starter charts

| Chart type | Table(s) | Dimension | Metric |
|---|---|---|---|
| Bar — group size distribution | `groups` | `member_count` | `COUNT(group_id)` |
| Bar — top rated movies by groups | `group_ratings` + `movies` | `title` | `COUNT(*)` where `rating = 1` |
| Pie — genre distribution | `movies` | `genre` | `COUNT(movie_id)` |
| Time series — rating activity | `group_ratings` | `timestamp` | `COUNT(*)` |
| Bar — ratings per split | `group_ratings` | `split` | `COUNT(*)` |

---

## Troubleshooting

**`bq: command not found`**
Run `exec -l $SHELL` to reload your shell after installation, or open a new terminal.

**`Access Denied`**
Make sure you ran `gcloud auth login` and that `gcloud config get-value project` returns the correct project ID.

**`Not found: Dataset group_recommender`**
You skipped Step 5 or used a different project. Run `bq mk --location=US group_recommender` again after checking `gcloud config get-value project`.

**Schema looks wrong (all columns detected as STRING)**
The autodetect can fail on large files occasionally. Add `--schema` to specify types explicitly, e.g.:
```bash
bq load --source_format=CSV \
    --schema 'group_id:INTEGER,movie_id:INTEGER,rating:INTEGER,timestamp:TIMESTAMP,split:STRING' \
    group_recommender.group_ratings \
    "$CSV_DIR/group_ratings.csv"
```
