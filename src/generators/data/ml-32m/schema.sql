-- ============================================================
-- MovieLens 32M Dataset -- MariaDB Schema
-- 32,000,204 ratings | 87,585 movies | 200,948 users
-- 2,000,072 tags
-- Source files: ratings.csv, movies.csv, tags.csv, links.csv
-- ============================================================

CREATE DATABASE IF NOT EXISTS grouprec
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE grouprec;

-- ─── users ───────────────────────────────────────────────────────────────────
-- Anonymous user IDs; no other attributes in the dataset.
-- Populated by extracting distinct IDs from ratings.csv (see load section).
CREATE TABLE IF NOT EXISTS users (
    user_id INT UNSIGNED NOT NULL,
    PRIMARY KEY (user_id)
) ENGINE=InnoDB;

-- ─── movies ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS movies (
    movie_id     INT UNSIGNED     NOT NULL,
    title        VARCHAR(500)     NOT NULL,
    release_year SMALLINT UNSIGNED,               -- parsed from "(YYYY)" suffix in title
    genres_raw   VARCHAR(300)     NOT NULL DEFAULT '(no genres listed)',  -- original pipe-separated value
    PRIMARY KEY (movie_id),
    KEY idx_movies_year (release_year)
) ENGINE=InnoDB
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- ─── genres (normalised) ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS genres (
    genre_id TINYINT UNSIGNED NOT NULL AUTO_INCREMENT,
    name     VARCHAR(50)      NOT NULL,
    PRIMARY KEY (genre_id),
    UNIQUE KEY uq_genre_name (name)
) ENGINE=InnoDB
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

INSERT IGNORE INTO genres (name) VALUES
    ('Action'),
    ('Adventure'),
    ('Animation'),
    ('Children'),
    ('Comedy'),
    ('Crime'),
    ('Documentary'),
    ('Drama'),
    ('Fantasy'),
    ('Film-Noir'),
    ('Horror'),
    ('Musical'),
    ('Mystery'),
    ('Romance'),
    ('Sci-Fi'),
    ('Thriller'),
    ('War'),
    ('Western'),
    ('(no genres listed)');

-- Junction table: one row per (movie, genre) pair
CREATE TABLE IF NOT EXISTS movie_genres (
    movie_id INT UNSIGNED     NOT NULL,
    genre_id TINYINT UNSIGNED NOT NULL,
    PRIMARY KEY (movie_id, genre_id),
    KEY idx_movie_genres_genre (genre_id),
    CONSTRAINT fk_mg_movie FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
    CONSTRAINT fk_mg_genre FOREIGN KEY (genre_id) REFERENCES genres  (genre_id)
) ENGINE=InnoDB;

-- ─── links ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS links (
    movie_id INT UNSIGNED NOT NULL,
    imdb_id  VARCHAR(10),       -- kept as string to preserve leading zeros (e.g. "0114709")
    tmdb_id  INT UNSIGNED,
    PRIMARY KEY (movie_id),
    CONSTRAINT fk_links_movie FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
) ENGINE=InnoDB;

-- ─── ratings ─────────────────────────────────────────────────────────────────
-- 32,000,204 rows; source file is ordered by (user_id, movie_id)
CREATE TABLE IF NOT EXISTS ratings (
    user_id  INT UNSIGNED  NOT NULL,
    movie_id INT UNSIGNED  NOT NULL,
    rating   DECIMAL(2,1)  NOT NULL,   -- 0.5 – 5.0 in half-star increments
    rated_at DATETIME      NOT NULL,   -- converted from Unix epoch timestamp
    PRIMARY KEY (user_id, movie_id),
    KEY idx_ratings_movie    (movie_id),
    KEY idx_ratings_rated_at (rated_at),
    CONSTRAINT fk_ratings_user  FOREIGN KEY (user_id)  REFERENCES users  (user_id),
    CONSTRAINT fk_ratings_movie FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
) ENGINE=InnoDB
  ROW_FORMAT=COMPRESSED;    -- ~30% space saving on this large table

-- Optional: distribute 32M rows across partitions for parallel scans.
-- Uncomment if your MariaDB instance has the partition storage engine enabled.
-- PARTITION BY HASH (user_id) PARTITIONS 8;

-- ─── tags ────────────────────────────────────────────────────────────────────
-- 2,000,072 rows
CREATE TABLE IF NOT EXISTS tags (
    tag_id    BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    user_id   INT UNSIGNED    NOT NULL,
    movie_id  INT UNSIGNED    NOT NULL,
    tag       VARCHAR(500)    NOT NULL,
    tagged_at DATETIME        NOT NULL,   -- converted from Unix epoch timestamp
    PRIMARY KEY (tag_id),
    KEY idx_tags_user  (user_id),
    KEY idx_tags_movie (movie_id),
    CONSTRAINT fk_tags_user  FOREIGN KEY (user_id)  REFERENCES users  (user_id),
    CONSTRAINT fk_tags_movie FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
) ENGINE=InnoDB
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;


-- ============================================================
-- DATA LOADING
-- Run from the directory that contains the CSV files, or
-- replace file names with absolute paths.
-- Requires FILE privilege and local-infile enabled:
--   mariadb --local-infile=1 -u <user> -p grouprec < schema.sql
-- ============================================================

-- Disable checks for fast bulk loading
SET foreign_key_checks   = 0;
SET unique_checks        = 0;
-- SET sql_log_bin = 0;  -- requires BINLOG ADMIN privilege; omit for non-root users
SET autocommit           = 0;

-- ── 1. movies ────────────────────────────────────────────────────────────────
-- Parses release_year from the "(YYYY)" suffix at the end of the title.
LOAD DATA LOCAL INFILE 'movies.csv'
    INTO TABLE movies
    FIELDS TERMINATED BY ','
           ENCLOSED BY '"'
    LINES  TERMINATED BY '\n'
    IGNORE 1 LINES
    (@movie_id, @title, @genres)
    SET movie_id     = @movie_id,
        title        = @title,
        genres_raw   = @genres,
        release_year = IF(
            @title REGEXP '\\([0-9]{4}\\)\\s*$',
            CAST(SUBSTR(@title, LENGTH(@title) - 4, 4) AS UNSIGNED),
            NULL
        );

COMMIT;

-- ── 2. Populate movie_genres from genres_raw ─────────────────────────────────
-- This stored procedure splits the pipe-separated genres_raw field and inserts
-- one row per genre into movie_genres.
DELIMITER $$

CREATE OR REPLACE PROCEDURE populate_movie_genres()
BEGIN
    DECLARE done      INT DEFAULT 0;
    DECLARE v_movie   INT UNSIGNED;
    DECLARE v_raw     VARCHAR(300);
    DECLARE v_genre   VARCHAR(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
    DECLARE v_pos     INT;
    DECLARE v_genre_id TINYINT UNSIGNED;

    DECLARE cur CURSOR FOR SELECT movie_id, genres_raw FROM movies;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN cur;
    read_loop: LOOP
        FETCH cur INTO v_movie, v_raw;
        IF done THEN LEAVE read_loop; END IF;

        SET v_raw = CONCAT(v_raw, '|');   -- sentinel for the last token
        WHILE LENGTH(v_raw) > 0 DO
            SET v_pos   = LOCATE('|', v_raw);
            SET v_genre = TRIM(LEFT(v_raw, v_pos - 1));
            SET v_raw   = SUBSTR(v_raw, v_pos + 1);

            IF LENGTH(v_genre) > 0 THEN
                SELECT genre_id INTO v_genre_id FROM genres WHERE name = v_genre LIMIT 1;
                IF v_genre_id IS NOT NULL THEN
                    INSERT IGNORE INTO movie_genres (movie_id, genre_id)
                    VALUES (v_movie, v_genre_id);
                END IF;
            END IF;
        END WHILE;
    END LOOP;
    CLOSE cur;
END$$

DELIMITER ;

CALL populate_movie_genres();
DROP PROCEDURE populate_movie_genres;

COMMIT;

-- ── 3. links ─────────────────────────────────────────────────────────────────
LOAD DATA LOCAL INFILE 'links.csv'
    INTO TABLE links
    FIELDS TERMINATED BY ','
           ENCLOSED BY '"'
    LINES  TERMINATED BY '\n'
    IGNORE 1 LINES
    (movie_id, imdb_id, @tmdb)
    SET tmdb_id = NULLIF(@tmdb, '');

COMMIT;

-- ── 4. users (derived from ratings) ──────────────────────────────────────────
-- There is no users.csv; user IDs are extracted from ratings.csv.
-- Load ratings into a staging table first, then derive users.
CREATE TEMPORARY TABLE ratings_stage (
    user_id  INT UNSIGNED NOT NULL,
    movie_id INT UNSIGNED NOT NULL,
    rating   DECIMAL(2,1) NOT NULL,
    ts       INT UNSIGNED NOT NULL
) ENGINE=InnoDB;

LOAD DATA LOCAL INFILE 'ratings.csv'
    INTO TABLE ratings_stage
    FIELDS TERMINATED BY ','
           ENCLOSED BY '"'
    LINES  TERMINATED BY '\n'
    IGNORE 1 LINES
    (user_id, movie_id, rating, ts);

INSERT IGNORE INTO users (user_id)
    SELECT DISTINCT user_id FROM ratings_stage;

COMMIT;

-- ── 5. ratings (from staging) ─────────────────────────────────────────────────
INSERT INTO ratings (user_id, movie_id, rating, rated_at)
    SELECT user_id, movie_id, rating, FROM_UNIXTIME(ts)
    FROM ratings_stage;

DROP TEMPORARY TABLE ratings_stage;

COMMIT;

-- ── 6. tags ──────────────────────────────────────────────────────────────────
-- Insert any new user IDs that appear only in tags (rare but possible)
CREATE TEMPORARY TABLE tags_stage (
    user_id  INT UNSIGNED NOT NULL,
    movie_id INT UNSIGNED NOT NULL,
    tag      VARCHAR(500) NOT NULL,
    ts       INT UNSIGNED NOT NULL
) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

LOAD DATA LOCAL INFILE 'tags.csv'
    INTO TABLE tags_stage
    FIELDS TERMINATED BY ','
           ENCLOSED BY '"'
    LINES  TERMINATED BY '\n'
    IGNORE 1 LINES
    (user_id, movie_id, tag, ts);

INSERT IGNORE INTO users (user_id)
    SELECT DISTINCT user_id FROM tags_stage
    WHERE user_id NOT IN (SELECT user_id FROM users);

INSERT INTO tags (user_id, movie_id, tag, tagged_at)
    SELECT user_id, movie_id, tag, FROM_UNIXTIME(GREATEST(ts, 1))
    FROM tags_stage;

DROP TEMPORARY TABLE tags_stage;

COMMIT;

-- Re-enable constraints
SET foreign_key_checks = 1;
SET unique_checks      = 1;
SET autocommit         = 1;
