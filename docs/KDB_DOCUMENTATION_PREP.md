# KDB+ Documentation Prep for RAG System

This document contains curated content from code.kx.com ready for ingestion into the Q for Mortals RAG system. Execute the scraper with these additional URLs to expand the knowledge base.

---

## Source URLs to Add to Scraper

### Reference Documentation (code.kx.com/q/ref/)
```
https://code.kx.com/q/ref/
https://code.kx.com/q/ref/aj/
https://code.kx.com/q/ref/apply/
https://code.kx.com/q/ref/amend/
https://code.kx.com/q/ref/attr/
https://code.kx.com/q/ref/cast/
https://code.kx.com/q/ref/cond/
https://code.kx.com/q/ref/delete/
https://code.kx.com/q/ref/dotq/
https://code.kx.com/q/ref/dotz/
https://code.kx.com/q/ref/each/
https://code.kx.com/q/ref/enlist/
https://code.kx.com/q/ref/enum-extend/
https://code.kx.com/q/ref/exec/
https://code.kx.com/q/ref/fby/
https://code.kx.com/q/ref/flip/
https://code.kx.com/q/ref/get/
https://code.kx.com/q/ref/group/
https://code.kx.com/q/ref/hopen/
https://code.kx.com/q/ref/if/
https://code.kx.com/q/ref/ij/
https://code.kx.com/q/ref/iterators/
https://code.kx.com/q/ref/key/
https://code.kx.com/q/ref/lj/
https://code.kx.com/q/ref/meta/
https://code.kx.com/q/ref/over/
https://code.kx.com/q/ref/read0/
https://code.kx.com/q/ref/read1/
https://code.kx.com/q/ref/select/
https://code.kx.com/q/ref/set/
https://code.kx.com/q/ref/string/
https://code.kx.com/q/ref/tables/
https://code.kx.com/q/ref/til/
https://code.kx.com/q/ref/type/
https://code.kx.com/q/ref/update/
https://code.kx.com/q/ref/upsert/
https://code.kx.com/q/ref/value/
https://code.kx.com/q/ref/wj/
https://code.kx.com/q/ref/xasc/
```

### Database Documentation (code.kx.com/q/database/)
```
https://code.kx.com/q/database/
https://code.kx.com/q/database/object/
https://code.kx.com/q/database/segment/
```

### Basics Documentation (code.kx.com/q/basics/)
```
https://code.kx.com/q/basics/datatypes/
https://code.kx.com/q/basics/syscmds/
https://code.kx.com/q/basics/cmdline/
https://code.kx.com/q/basics/qsql/
https://code.kx.com/q/basics/joins/
```

### Knowledge Base (code.kx.com/q/kb/)
```
https://code.kx.com/q/kb/file-compression/
```

---

## Extracted Content

### 1. DATATYPES (code.kx.com/q/basics/datatypes/)

| Type | Code | Name | Size | Literal | Null | Infinity |
|------|------|------|------|---------|------|----------|
| Boolean | b | boolean | 1 byte | 0b | - | - |
| Guid | g | guid | 16 bytes | 0Ng | - | - |
| Byte | x | byte | 1 byte | 0x00 | - | - |
| Short | h | short | 2 bytes | 0h | 0Nh | 0Wh |
| Int | i | int | 4 bytes | 0i | 0Ni | 0Wi |
| Long | j | long | 8 bytes | 0j | 0Nj | 0Wj |
| Real | e | real | 4 bytes | 0e | 0Ne | 0We |
| Float | f | float | 8 bytes | 0.0 | 0n | 0w |
| Char | c | char | 1 byte | " " | - | - |
| Symbol | s | symbol | variable | ` | - | - |
| Timestamp | p | timestamp | 8 bytes | dateDtimespan | 0Np | 0Wp |
| Month | m | month | 4 bytes | 2000.01m | 0Nm | 0Wm |
| Date | d | date | 4 bytes | 2000.01.01 | 0Nd | 0Wd |
| DateTime | z | datetime | 8 bytes | dateTtime | 0Nz | 0wz |
| Timespan | n | timespan | 8 bytes | 00:00:00.000000000 | 0Nn | 0Wn |
| Minute | u | minute | 4 bytes | 00:00 | 0Nu | 0Wu |
| Second | v | second | 4 bytes | 00:00:00 | 0Nv | 0Wv |
| Time | t | time | 4 bytes | 00:00:00.000 | 0Nt | 0Wt |

**Key Concepts:**
- Default integer type is long (`7h` or `"j"`) as of V3.0
- No dedicated string type; character vectors (type `10h`) serve this purpose
- Symbols use backtick prefix (`` ` ``)
- Valid dates span from `0001.01.01` to `9999.12.31`
- Extended types: Enumerated (20h-76h), Nested (77-96), Dictionary (99h), Table (98h), Functions/Lambdas (100-112h)
- Datetime type (15) is deprecated in favor of timestamp (12)

---

### 2. REFERENCE CARD STRUCTURE (code.kx.com/q/ref/)

**Keywords by Category:**
- **Control**: `do`, `exit`, `if`, `while`
- **Environment**: `getenv`, `gtime`, `ltime`, `setenv`
- **I/O**: `dsave`, `get`, `hclose`, `hcount`, `hdel`, `hopen`, `hsym`, `load`, `read0`, `read1`, `rload`, `rsave`, `save`, `set`
- **Iteration**: `each`, `over`, `peach`, `prior`, `scan`
- **Joins**: `aj`, `ajf`, `asof`, `ej`, `ij`, `lj`, `pj`, `uj`, `wj`
- **List operations**: `count`, `cross`, `cut`, `enlist`, `except`, `fills`, `first`, `flip`, `group`, `in`, `inter`, `last`, `next`, `prev`, `raze`, `reverse`, `rotate`, `sublist`, `til`, `union`, `where`
- **Math**: `abs`, `acos`, `asin`, `atan`, `avg`, `ceiling`, `cos`, `div`, `exp`, `floor`, `log`, `max`, `min`, `mod`, `neg`, `sqrt`, `tan`, `var`
- **Query**: `delete`, `exec`, `fby`, `select`, `update`
- **Sort**: `asc`, `bin`, `desc`, `distinct`, `rank`, `xbar`, `xrank`
- **Text**: `like`, `lower`, `ltrim`, `md5`, `rtrim`, `ss`, `ssr`, `string`, `trim`, `upper`
- **Table operations**: `cols`, `csv`, `fkeys`, `insert`, `key`, `keys`, `meta`, `ungroup`, `upsert`, `xasc`, `xcol`, `xcols`, `xdesc`, `xgroup`, `xkey`

**Operators:**
- `+`, `-`, `*`, `%` - Arithmetic
- `.`, `@`, `$`, `!`, `?` - Overloaded operators
- `=`, `<>`, `~`, `<`, `<=`, `>=`, `>` - Comparison
- `|`, `&` - Logical
- `#`, `_`, `:`, `^`, `,` - Structural

**Iterators:** `'`, `/:`, `/`, `\:`, `\`, `':` with `each`, `over`, `scan`, `prior`, `peach`

**Attributes:** `g` (grouped), `p` (parted), `s` (sorted), `u` (unique)

---

### 3. DATABASE - TABLES IN FILESYSTEM (code.kx.com/q/database/)

kdb+ represents tables and columns in the filesystem as eponymous directories and binary files.

**Four serialization approaches:**
1. **Object**: Single binary file format
2. **Splayed table**: Directory of column files
3. **Partitioned table**: Date/time-divided records
4. **Segmented database**: Multi-disk distribution

**Topics covered:**
- Populating tables (loading from large files, foreign keys, linking columns, data loaders)
- Persisting tables (serializing objects, splayed table management, partitioned table setup)
- Maintenance (data management, encryption, compression, access permissions, query optimization)

---

### 4. SERIALIZING OBJECTS (code.kx.com/q/database/object/)

**`save` and `load`:**
```q
save `cities        / writes to cities file
load `cities        / reads from cities file
```

**`set` and `get`:**
```q
`:foo/bar/bigcities set cities    / write to specific path
get `:foo/bar/bigcities           / retrieve table
```

Serialization survives the session's sym list - enumerations and foreign keys persist correctly.

---

### 5. SEGMENTED DATABASES (code.kx.com/q/database/segment/)

Segmented databases distribute partitioned tables across multiple storage devices.

**par.txt Configuration:**
Each row is a directory path. Example:
```
/1/db
/2/db
```

**Multithreading:** Partition `p` is assigned to secondary thread `p mod n`, maximizing parallelization.

**Key considerations:**
- Each directory must be non-empty
- No trailing folder delimiters in paths
- Use `.Q.par` to verify partition locations

---

### 6. AS-OF JOIN (code.kx.com/q/ref/aj/)

```q
aj  [c; t1; t2]
aj0 [c; t1; t2]
ajf [c; t1; t2]
ajf0[c; t1; t2]
```

Performs left-join combining records from two tables. Matches equality on columns `c[til n-1]` and uses most recent value of final column `c[n]`.

**Example:**
```q
q)t:([]time:10:01:01 10:01:03 10:01:04;sym:`msft`ibm`ge;qty:100 200 150)
q)q:([]time:10:01:00 10:01:00 10:01:00 10:01:02;sym:`ibm`msft`msft`ibm;px:100 99 101 98)
q)aj[`sym`time;t;q]
time       sym  qty px
---------------------
10:01:01 msft 100 101
10:01:03 ibm  200 98
10:01:04 ge   150
```

**Variants:**
- `aj`: Returns boundary time from t1
- `aj0`: Returns actual time from t2
- `ajf/ajf0`: Fill from LHS if RHS is null

**Performance:** Should run at a million or two trade records per second.

---

### 7. INNER JOIN (code.kx.com/q/ref/ij/)

```q
x ij y      ij[x;y]
x ijf y     ijf[x;y]
```

Returns two tables joined on key columns. Result has one combined record for each row in `x` that matches a row in `y`.

- Common columns: Values from y replace those in x
- `ijf` variant: V2.8 compatibility (preserves x values when y has nulls)
- Multithreaded primitive

---

### 8. LEFT JOIN (code.kx.com/q/ref/lj/)

```q
x lj y      lj[x;y]
x ljf y     ljf[x;y]
```

Each record in x is matched with records from y based on key columns:
- If matching record in y: joined to x record, common columns replaced
- If no matching record: common columns unchanged, new columns are null

---

### 9. WINDOW JOIN (code.kx.com/q/ref/wj/)

```q
wj [w; c; t; (q; (f0;c0); (f1;c1))]
wj1[w; c; t; (q; (f0;c0); (f1;c1))]
```

For each record in table `t`, returns enriched record with aggregation results from matching intervals.

**wj vs wj1:**
- `wj`: Prevailing quote on entry to window is valid
- `wj1`: Only considers quotes on or after entry to window

---

### 10. SELECT STATEMENT (code.kx.com/q/ref/select/)

```q
select [Lexp] [ps] [by pb] from texp [where pw]
```

**Components:**
- `Lexp`: Limit expression (`select[n]`, `select[m n]`, `select[order]`)
- `ps`: Select phrase (columns/expressions)
- `pb`: By phrase (grouping)
- `texp`: Table expression
- `pw`: Where conditions

**Features:**
- `select distinct` returns unique records
- By phrase columns automatically appear in results
- Cond unsupported within query templates

---

### 11. EXEC STATEMENT (code.kx.com/q/ref/exec/)

```q
exec [distinct] ps [by pb] from texp [where pw]
```

**Return types:**
- Omitted select phrase: returns last record
- Single column: returns list of values
- Multiple columns: returns dictionary
- With by clause: returns grouped dictionary

---

### 12. UPDATE STATEMENT (code.kx.com/q/ref/update/)

```q
update <select_phrase> [by <by_phrase>] from <table_expression> [where <where_phrase>]
```

Will not modify a splayed table on disk. New columns get nulls for unmatched rows.

---

### 13. DELETE STATEMENT (code.kx.com/q/ref/delete/)

```q
delete    from x
delete    from x where pw
delete ps from x
```

Removes rows, columns, dictionary entries, or namespace objects.

---

### 14. UPSERT (code.kx.com/q/ref/upsert/)

```q
x upsert y    upsert[x;y]
```

For keyed tables: updates matching keys, inserts non-matching records.
For simple tables: appends records.
For splayed tables: appends to column files directly.

---

### 15. ITERATORS (code.kx.com/q/ref/iterators/)

**Maps:** Distribute function application across list/dictionary items with implicit parallelism.
**Accumulators:** Apply functions successively (fold/reduce operations).

**Core Iterators:**
- **Each (`'`)**: `(u')x` or `u each x`
- **Over (`/`)**: `(b/)y` or `b over y` - reduce to single value
- **Scan (`\`)**: `(g\)y` or `g scan y` - returns intermediate results
- **Each Prior (`':`)**: `b prior y` - apply to successive pairs
- **Each Parallel (`peach`)**: Parallel execution across items
- **Each Left/Right (`\:`, `/:`)**: Apply with fixed argument

---

### 16. .Q NAMESPACE (code.kx.com/q/ref/dotq/)

**Key utilities:**
- **General**: String manipulation (`.Q.A`, `.Q.a`, `.Q.an`), encoding (`.Q.btoa`, `.Q.atob`)
- **Database**: Saving/loading (`.Q.dpft`, `.Q.dpt`, `.Q.en`, `.Q.ld`)
- **Partitioned DB State**: `.Q.D`, `.Q.P`, `.Q.PV`, `.Q.PD`
- **Debug/Profile**: `.Q.bt`, `.Q.sbt`, `.Q.trp`, `.Q.prf0`
- **HTTP**: `.Q.hg` (GET), `.Q.hp` (POST)
- **File streaming**: `.Q.fs`, `.Q.fsn`
- **Memory**: `.Q.gc` (garbage collection)

---

### 17. .z NAMESPACE (code.kx.com/q/ref/dotz/)

**Environment Variables:**
- Time/Date: `.z.Z`, `.z.z`, `.z.P`, `.z.p`, `.z.N`, `.z.n`, `.z.T`, `.z.t`, `.z.D`, `.z.d`
- System: `.z.K` (version), `.z.k` (release date), `.z.o` (OS), `.z.c` (cores), `.z.i` (PID), `.z.h` (hostname)
- Connection: `.z.u` (user), `.z.w` (handle), `.z.W` (IPC handles)

**Callbacks:**
- Connection: `.z.po` (open), `.z.pc` (close), `.z.pw` (validation)
- Message: `.z.pg` (sync get), `.z.ps` (async set), `.z.pi` (console input)
- HTTP: `.z.ph` (GET), `.z.pp` (POST), `.z.pm` (other methods)
- WebSocket: `.z.wo`, `.z.ws`, `.z.wc`
- Timer: `.z.ts`
- Config: `.z.zd` (compression defaults)

---

### 18. APPLY OPERATOR (code.kx.com/q/ref/apply/)

**Binary Forms:**
```q
(+) . 2 3    / returns 5
.[+;2 3]     / equivalent form
```

**Trap (error handling):**
```q
.[g;gx;e]    / attempts g . gx, on error evaluates e
@[f;fx;e]    / unary trap form
```

**Example:**
```q
.[+;"ab";`ouch]           / returns `ouch
.[+;"ab";{"Wrong ",x}]    / returns "Wrong type"
```

---

### 19. AMEND OPERATOR (code.kx.com/q/ref/amend/)

**Ternary:**
```q
.[d; i; u]    @[d; i; u]
```

**Quaternary:**
```q
.[d; i; v; vy]    @[d; i; v; vy]
```

Modifies items in a list, dictionary or datafile.

---

### 20. CAST OPERATOR (code.kx.com/q/ref/cast/)

```q
x$y     $[x;y]
```

Converts data to another datatype by reinterpreting bit pattern.

**Datatype codes:**
- `1h/"b"`: boolean
- `6h/"i"`: int
- `7h/"j"`: long
- `9h/"f"`: float
- `14h/"d"`: date
- `16h/"n"`: timespan
- `19h/"t"`: time

Temporal casting truncates (uses floor logic), not rounds.

---

### 21. VALUE / GET (code.kx.com/q/ref/value/)

```q
value x     value[x]
get x       get[x]
```

Recursively evaluates expressions and retrieves stored values.

**Return types by input:**
- Dictionary: returns values
- Symbol: returns variable's value
- String: evaluates in current context
- List: calls/indexes first element with remaining elements
- Lambda: returns structure including bytecode

---

### 22. SET (code.kx.com/q/ref/set/)

```q
`a set 42                    / global variable
`:a set 42                   / serialize to file
`:tbl/ set t                 / splay table
(`:ztbl;17;2;6) set t       / compressed file
```

Symbol columns must be enumerated before splaying.

---

### 23. FILE COMPRESSION (code.kx.com/q/kb/file-compression/)

```q
(`:filename;logicalBlockSize;algorithm;level) set data
```

**Algorithms:**
| Alg | Description | Levels |
|-----|-------------|--------|
| 0 | None | 0 |
| 1 | q IPC | 0 |
| 2 | Gzip | 0-9 |
| 3 | Snappy | 0 |
| 4 | LZ4HC | 0-16 |
| 5 | Zstd | -7 to 22 |

**Default compression:**
```q
.z.zd:17 2 6  / logicalBlockSize, algorithm, level
```

**Restrictions:** Don't compress log files or nested column companion files.

---

### 24. SYSTEM COMMANDS (code.kx.com/q/basics/syscmds/)

**Information:** `\a` (tables), `\b` (dependencies), `\f` (functions), `\v` (variables), `\w` (memory)
**Configuration:** `\c` (console size), `\d` (namespace), `\o` (UTC offset), `\P` (precision)
**Performance:** `\g` (garbage collection), `\s` (secondary threads), `\t` (timer), `\ts` (time/space)
**File/Data:** `\l` (load), `\cd` (change directory)
**Session:** `\p` (port), `\\` (exit)

---

### 25. COMMAND LINE OPTIONS (code.kx.com/q/basics/cmdline/)

```
q [file] [-option [parameters] ...]
```

**Key options:**
- `-b`: Blocked (restrict write-access)
- `-c r c`: Console size
- `-e [0|1|2]`: Error traps
- `-g [0|1]`: Garbage collection mode
- `-l`: Log updates
- `-p`: Listening port
- `-s N`: Secondary threads
- `-t N`: Timer ticks (ms)
- `-T N`: Timeout (seconds)
- `-u 1`: Disable syscmds
- `-w N`: Workspace limit (MB)

---

### 26. qSQL (code.kx.com/q/basics/qsql/)

**Templates:**
- `select` - returns table parts
- `exec` - returns columns as lists/dictionaries
- `update` - adds or modifies rows/columns
- `delete` - removes rows or columns

**Evaluation order:** from -> where -> by -> select -> limit

**Virtual Column:** `i` represents row indices

**Performance tips:**
- Select only required columns
- Place most restrictive constraints first
- Ensure appropriate attributes on constrained columns

---

### 27. JOINS OVERVIEW (code.kx.com/q/basics/joins/)

**Keyed Joins:**
- Coalesce (`^`): Merges ignoring nulls
- Equi join (`ej`): Specify match columns
- Inner join (`ij`): Match on key columns
- Left join (`lj`): Outer join on keys
- Plus join (`pj`): Adds values instead of replacing
- Union join (`uj`): Uses all rows from both tables

**As-of Joins:**
- Window join (`wj`): Aggregate values in time intervals
- As-of join (`aj`): Uses last value per interval

**Implicit Joins:** Foreign keys enable automatic joining

---

### 28. CORE FUNCTIONS

**enlist**: Creates list from arguments
```q
enlist 10      / atom to single-item list
enlist[a;b;c]  / multiple arguments
```

**flip**: Transposes lists, converts dictionary<->table
```q
flip (1 2 3;4 5 6)   / transpose
flip D               / dictionary to table
```

**group**: Maps distinct items to positions
```q
group "mississippi"  / m->,0; i->1 4 7 10; s->2 3 5 6; p->8 9
```

**til**: First x natural numbers
```q
til 5   / 0 1 2 3 4
```

**key**: Dictionary keys, file existence, variable check, type info

**meta**: Table metadata (column, type, foreign key, attribute)

**type**: Returns datatype as short integer

**attr**: Returns attributes (s/u/p/g) as symbol vector

---

### 29. CONTROL FLOW

**Cond (`$`):**
```q
$[test;et;ef;...]   / lazy evaluation
$[q;a;r;b;c]        / chained conditions
```

**if:**
```q
if[test;e1;e2;...;en]   / conditional execution, returns null
```

**over/scan:**
```q
v1 over x    / converge or reduce
v1 scan x    / returns intermediate results
```

---

### 30. I/O FUNCTIONS

**read0**: Read text from files
```q
read0 f           / all lines as string list
read0 (f;o;n)     / n chars from offset o
```

**read1**: Read bytes from files
```q
read1 f           / entire file as bytes
read1 (f;o;n)     / n bytes from offset o
```

**hopen**: Establish connections
```q
hopen `:host:port           / TCP
hopen `:unix://port         / Unix socket
hopen `:tcps://host:port    / TLS
```

---

## Execution Plan

1. **Modify `src/scraper/discover.py`** to include these additional URL patterns
2. **Run scraper** to fetch new documentation
3. **Run embedding pipeline** to generate vectors
4. **Test queries** to verify new knowledge is accessible

The above content has been extracted and formatted for direct use in your RAG system.
