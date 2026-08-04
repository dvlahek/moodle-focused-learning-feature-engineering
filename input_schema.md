# Input schema

The package accepts an English canonical schema and common Croatian Moodle-export labels.

## Moodle event log

| Canonical column | Accepted Croatian alias | Type | Description |
|---|---|---|---|
| `student_id` | `Puno ime` | text | Student identifier. Use anonymized identifiers. |
| `timestamp` | `Time` | datetime | Event timestamp before the assessment. |
| `component` | `Komponenta` | text | Moodle component, such as File, Assignment, Quiz, or Forum. |
| `context` | `Kontekst` | text | Resource or activity context. |
| `event_name` | `Naziv` | text | Moodle event description. |

## Grade file

| Column | Type | Description |
|---|---|---|
| `student_id` | text | Identifier matching the event log. |
| `kolokvij1` | numeric | First-colloquium score on the 0-25 scale. |

The code also recognizes the target aliases `Kolokvij 1. (12,5/25)`, `Kolokvij 1`, `kolokvij 1`, `kolokvij_1`, and `K1`.

## Environmental file

| Column | Type | Description |
|---|---|---|
| `date` | date | Daily observation date. |
| `pm10`, `pm2_5` | numeric | Particulate-matter concentrations. |
| `tavg`, `tmin`, `tmax` | numeric | Daily temperatures. |
| `prcp` | numeric | Daily precipitation. |
| `wdir`, `wspd` | numeric | Wind direction and speed. |

All files must be CSV. The example files under `data_dummy/` are entirely synthetic.

