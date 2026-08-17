# Input schema

The supplementary package accepts an English canonical schema and common Croatian Moodle-export labels. The public example files under `data_dummy/` are entirely synthetic.

## Moodle event log

| Canonical column | Common Croatian alias | Type | Description |
|---|---|---|---|
| `student_id` | `Puno ime` | text | Student identifier. Use anonymized identifiers. |
| `timestamp` | `Time` | datetime | Event timestamp. Only events before the assessment are used. |
| `component` | `Komponenta` | text | Moodle component, such as File, Assignment, Quiz/Test, or Forum. |
| `context` | `Kontekst` | text | Resource/activity context. |
| `event_name` | `Naziv` | text | Moodle event description. |

The study-specific learning-material filter recognizes PDF contexts corresponding to the first eight instructional materials, including Croatian labels such as `Datoteka: 1 ... Datoteka: 8` and the equivalent synthetic English labels `Learning material 1 ... Learning material 8`.

## Assessment/course-record file

| Column | Type | Description |
|---|---|---|
| `student_id` | text | Identifier matching the event log. |
| `kolokvij1` | numeric | First-colloquium score on the 0–25 scale. |
| `Vjezbe` | numeric, optional | Exercise-related activity/participation variable when consistently available. |

The target aliases `Kolokvij 1. (12,5/25)`, `Kolokvij 1`, `kolokvij 1`, `kolokvij_1`, and `K1` are recognized. The Croatian spelling `Vježbe` is normalized to `Vjezbe`.

Only the explicitly supported columns above are imported from the assessment/course-record file. Other grade-book columns are ignored so that unrelated assessment information cannot silently enter the predictive feature matrix.

## Environmental file

| Column | Type | Description |
|---|---|---|
| `date` | date | Daily observation date. |
| `pm10`, `pm2_5` | numeric | Particulate-matter indicators. |
| `tavg`, `tmin`, `tmax` | numeric | Daily temperature measures. |
| `prcp` | numeric | Daily precipitation. |
| `snow` | numeric, optional | Daily snow indicator/amount. |
| `wdir`, `wspd` | numeric | Wind direction and speed. |
| `wpgt` | numeric, optional | Peak wind gust. |
| `pres` | numeric, optional | Atmospheric pressure. |

The code uses every supported environmental column that is present in the input file and skips supported columns that are absent.

## Validation and privacy

All files must be CSV. Real student-level Moodle logs, identifiers, assessment records, and outputs derived from the protected study data must not be committed to the public repository. The included `data_dummy/` files are artificial and exist only to demonstrate the expected structure and test the workflow.
