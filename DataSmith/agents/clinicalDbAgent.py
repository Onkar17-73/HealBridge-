import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


try:
	import psycopg
	from psycopg import sql
except Exception as e:
	psycopg = None  # type: ignore
	sql = None  # type: ignore


@dataclass
class DischargeSummary:
	patient_name: str
	discharge_date: Optional[str]
	primary_diagnosis: Optional[str]
	medications: Optional[List[str]]
	dietary_restrictions: Optional[str]
	follow_up: Optional[str]
	warning_signs: Optional[str]
	discharge_instructions: Optional[str]


def load_env() -> None:
	# Load environment variables from .env if present
	load_dotenv(override=False)


def get_conn_params() -> Dict[str, Any]:
	params = {
		"host": os.getenv("PGHOST", "localhost"),
		"port": int(os.getenv("PGPORT", "5432")),
		"dbname": os.getenv("PGDATABASE"),
		"user": os.getenv("PGUSER"),
		"password": os.getenv("PGPASSWORD"),
		"sslmode": os.getenv("PGSSLMODE", "prefer"),
	}
	missing = [k for k, v in params.items() if k in {"dbname", "user", "password"} and not v]
	if missing:
		raise RuntimeError(
			"Missing DB credentials: "
			+ ", ".join(missing)
			+ ". Set PGDATABASE, PGUSER, PGPASSWORD (and optionally PGHOST, PGPORT, PGSSLMODE)."
		)
	return params


def sanitize_identifier(identifier: str) -> str:
	# Allow only simple identifiers (letters, numbers, underscore). Prevent SQL injection via identifiers.
	import re

	if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
		raise ValueError(f"Invalid identifier: {identifier}")
	return identifier


def fetch_by_name(
	name: str,
	table: str,
	schema: str = "public",
	limit: int = 10,
	match: str = "ilike",
) -> List[DischargeSummary]:
	if psycopg is None or sql is None:
		raise RuntimeError(
			"psycopg is not installed. Please run: pip install 'psycopg[binary]>=3.1'"
		)

	table = sanitize_identifier(table)
	schema = sanitize_identifier(schema)
	if match not in {"exact", "ilike"}:
		raise ValueError("match must be one of: exact, ilike")

	params = get_conn_params()

	query = sql.SQL(
		"""
		SELECT 
			patient_name,
			discharge_date::text,
			primary_diagnosis,
			medications,
			dietary_restrictions,
			follow_up,
			warning_signs,
			discharge_instructions
		FROM {schema}.{table}
		WHERE {name_clause}
		ORDER BY discharge_date DESC NULLS LAST
		LIMIT %s
		"""
	).format(
		schema=sql.Identifier(schema),
		table=sql.Identifier(table),
		name_clause=sql.SQL("patient_name = %s") if match == "exact" else sql.SQL("patient_name ILIKE %s"),
	)

	name_param = name if match == "exact" else f"%{name}%"

	results: List[DischargeSummary] = []
	with psycopg.connect(**params) as conn:
		with conn.cursor() as cur:
			cur.execute(query, (name_param, limit))
			rows = cur.fetchall()

	for r in rows:
		# medications might come as list or string
		meds = r[3]
		if isinstance(meds, str):
			# naive split fallback
			meds_list = [m.strip() for m in meds.split(",") if m.strip()]
		else:
			meds_list = meds
		results.append(
			DischargeSummary(
				patient_name=r[0],
				discharge_date=r[1],
				primary_diagnosis=r[2],
				medications=meds_list,
				dietary_restrictions=r[4],
				follow_up=r[5],
				warning_signs=r[6],
				discharge_instructions=r[7],
			)
		)

	return results


def main():
	parser = argparse.ArgumentParser(
		description="Clinical DB Agent: retrieve discharge summaries by patient name"
	)
	parser.add_argument("--name", required=True, help="Patient name or substring")
	parser.add_argument("--table", default="patient_discharges", help="Target table name")
	parser.add_argument("--schema", default="public", help="Schema name")
	parser.add_argument("--limit", type=int, default=10, help="Max rows to return")
	parser.add_argument("--match", choices=["exact", "ilike"], default="ilike", help="Name match mode")
	parser.add_argument("--json_only", action="store_true", help="Print JSON only (no extra text)")

	args = parser.parse_args()

	load_env()
	try:
		results = fetch_by_name(
			name=args.name,
			table=args.table,
			schema=args.schema,
			limit=args.limit,
			match=args.match,
		)
	except Exception as e:
		if args.json_only:
			print(json.dumps({"error": str(e)}))
		else:
			print(f"Error: {e}")
		sys.exit(1)

	data = [asdict(r) for r in results]
	if args.json_only:
		print(json.dumps(data, ensure_ascii=False, indent=2))
		return

	print(f"Found {len(data)} results\n")
	for i, row in enumerate(data, start=1):
		print(f"{i}. {row['patient_name']} | {row['discharge_date']} | {row['primary_diagnosis']}")
		print(f"   Meds: {row['medications']}")
		print(f"   Diet: {row['dietary_restrictions']}")
		print(f"   Follow-up: {row['follow_up']}")
		print(f"   Warning signs: {row['warning_signs']}")
		print(f"   Instructions: {row['discharge_instructions']}")
		print("-")


if __name__ == "__main__":
	main()
