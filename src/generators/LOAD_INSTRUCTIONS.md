# Loading MovieLens 32M into MariaDB

You only need **one file**: `schema.sql`. It creates all tables and loads all CSVs.

---

## Step 1 — Open a terminal in this folder

```bash
cd /home/bohdan/Projects/IAC-Group-Recommender/src/data/ml-32m
```

---

## Step 2 — Run the import

```bash
mariadb --local-infile=1 -h YOUR_SERVER_IP -u YOUR_USER -p < schema.sql
```

Replace `YOUR_SERVER_IP` with your server's IP address (e.g. `192.168.1.100`).  
Replace `YOUR_USER` with your MariaDB username (e.g. `root`).  
You will be prompted for your password.

That's it. The script creates the `grouprec` database, all tables, and loads the data automatically.

---

## Step 3 — Verify (optional)

Connect to MariaDB and check row counts:

```bash
mariadb -h YOUR_SERVER_IP -u YOUR_USER -p
```

Then paste:

```sql
USE grouprec;
SELECT 'movies',  COUNT(*) FROM movies
UNION ALL
SELECT 'ratings', COUNT(*) FROM ratings
UNION ALL
SELECT 'tags',    COUNT(*) FROM tags
UNION ALL
SELECT 'links',   COUNT(*) FROM links;
```

Expected:

| table   | rows       |
|---------|------------|
| movies  | 87,585     |
| ratings | 32,000,204 |
| tags    | 2,000,072  |
| links   | 87,585     |

---

## If you get an error

**`ERROR 2068: LOAD DATA LOCAL INFILE is disabled`**  
Run this once as your MariaDB admin user, then retry Step 2:
```sql
SET GLOBAL local_infile = 1;
```

**`ERROR 1045: Access denied`**  
Your user doesn't have enough privileges. Ask your DB admin or use the `root` user.

---

> The ratings table has 32 million rows — loading it takes **5–20 minutes** on a typical machine. That's normal.
