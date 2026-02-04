from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

from flask import Flask, render_template, request
import pandas as pd
import json
import os
import urllib.request

app = Flask(__name__)


def _numeric_series(series):
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _find_debit_credit_columns(df):
    name_map = {col: "".join(ch for ch in col.lower() if ch.isalnum()) for col in df.columns}
    numeric_counts = {
        col: _numeric_series(df[col]).notna().sum() for col in df.columns
    }

    debit_tokens = ("debit", "withdrawal", "dr", "outflow", "paid")
    credit_tokens = ("credit", "deposit", "cr", "inflow", "received")

    debit_candidates = [
        col for col, norm in name_map.items() if any(tok in norm for tok in debit_tokens)
    ]
    credit_candidates = [
        col for col, norm in name_map.items() if any(tok in norm for tok in credit_tokens)
    ]

    debit_col = max(debit_candidates, key=lambda c: numeric_counts.get(c, 0), default=None)
    credit_col = max(credit_candidates, key=lambda c: numeric_counts.get(c, 0), default=None)

    return debit_col, credit_col


def _opening_balance_mask(df):
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        if df[col].dtype == object:
            mask = mask | df[col].astype(str).str.contains("opening balance", case=False, na=False)
    return mask


def _normalize_amount(df, statement_type):
    debit_col, credit_col = _find_debit_credit_columns(df)
    if not debit_col or not credit_col:
        return df, "Could not detect debit and credit columns."

    debit_vals = _numeric_series(df[debit_col]).fillna(0)
    credit_vals = _numeric_series(df[credit_col]).fillna(0)

    if statement_type == "bank":
        amount = credit_vals - debit_vals
    else:
        amount = debit_vals - credit_vals

    amount = amount.where(~(_numeric_series(df[debit_col]).isna() & _numeric_series(df[credit_col]).isna()))

    open_mask = _opening_balance_mask(df)
    amount = amount.mask(open_mask)

    df = df.copy()
    df["amount"] = amount
    return df, None


def _read_preview(upload, statement_type):
    if not upload or upload.filename == "":
        return None, "No file uploaded."

    filename = upload.filename.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(upload)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(upload)
        elif filename.endswith(".pdf"):
            try:
                tables = pd.read_html(upload)
                df = tables[0] if tables else pd.DataFrame()
            except Exception:
                return None, "Could not extract a table from the PDF."
        else:
            return None, "Unsupported file type."
    except Exception:
        return None, "Failed to read file."

    try:
        normalized, normalize_error = _normalize_amount(df, statement_type)
    except Exception:
        return None, "Failed to process file."

    if normalize_error:
        return None, normalize_error

    preview = normalized.head(5)
    return preview, None


def _read_full(upload, statement_type):
    if not upload or upload.filename == "":
        return None, "No file uploaded."

    filename = upload.filename.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(upload)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(upload)
        elif filename.endswith(".pdf"):
            try:
                tables = pd.read_html(upload)
                df = tables[0] if tables else pd.DataFrame()
            except Exception:
                return None, "Could not extract a table from the PDF."
        else:
            return None, "Unsupported file type."
    except Exception:
        return None, "Failed to read file."

    try:
        normalized, normalize_error = _normalize_amount(df, statement_type)
    except Exception:
        return None, "Failed to process file."

    if normalize_error:
        return None, normalize_error

    return normalized, None


def _load_env_from_file(path=".env"):
    if os.getenv("OPENROUTER_API_KEY"):
        return
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def _call_reconciliation_agent(bank_df, ledger_df):
    _load_env_from_file()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None, "OPENROUTER_API_KEY is not set."

    bank_records = bank_df.where(pd.notna(bank_df), None).to_dict(orient="records")
    ledger_records = ledger_df.where(pd.notna(ledger_df), None).to_dict(orient="records")

    system_msg = (
        "You are an accounting reconciliation assistant. Use human-like reasoning to "
        "classify bank and ledger transactions as Matched, Partially matched, or Unmatched. "
        "Use description similarity (semantic, not exact), amount equality using absolute values, "
        "and date proximity within +/- 2 days. "
        "Apply these accounting rules: "
        "Bank inflow (+amount) aligns with Ledger debit (+amount). "
        "Bank outflow (-amount) aligns with Ledger credit (-amount). "
        "Provide plain-English explanations and required actions. "
        "Do not reveal hidden thoughts. Output clear, human-readable results."
    )

    user_msg = {
        "bank_transactions": bank_records,
        "ledger_transactions": ledger_records,
        "required_output": (
            "For each transaction, include: classification (Matched/Partially matched/Unmatched), "
            "explanation, and any required action. Use plain English."
        ),
    }

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg)},
        ],
        "temperature": 0.2,
    }

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        return content, None
    except Exception:
        return None, "AI reconciliation failed."


@app.route("/", methods=["GET", "POST"])
def index():
    bank_preview = None
    ledger_preview = None
    bank_error = None
    ledger_error = None
    reconciliation_summary = None
    reconciliation_error = None
    unmatched_summary = None
    password_error = None

    app_password = os.getenv("APP_PASSWORD")

    if request.method == "POST":
        if app_password:
            provided_password = None
            if request.authorization:
                provided_password = request.authorization.password
            if not provided_password:
                provided_password = request.form.get("app_password") or request.args.get("app_password")
            if provided_password != app_password:
                password_error = "Invalid password."
                return render_template(
                    "index.html",
                    bank_preview=bank_preview,
                    ledger_preview=ledger_preview,
                    bank_error=bank_error,
                    ledger_error=ledger_error,
                    reconciliation_summary=reconciliation_summary,
                    reconciliation_error=reconciliation_error,
                    unmatched_summary=unmatched_summary,
                    password_error=password_error,
                ), 403
        try:
            bank_file = request.files.get("bank_statement")
            ledger_file = request.files.get("ledger_statement")

            bank_preview, bank_error = _read_preview(bank_file, "bank")
            ledger_preview, ledger_error = _read_preview(ledger_file, "ledger")

            try:
                if bank_file:
                    bank_file.seek(0)
            except Exception:
                bank_error = bank_error or "Failed to process file."
            try:
                if ledger_file:
                    ledger_file.seek(0)
            except Exception:
                ledger_error = ledger_error or "Failed to process file."

            if not bank_error and not ledger_error and bank_file and ledger_file:
                bank_full, bank_full_error = _read_full(bank_file, "bank")
                ledger_full, ledger_full_error = _read_full(ledger_file, "ledger")
                if bank_full_error:
                    reconciliation_error = bank_full_error
                elif ledger_full_error:
                    reconciliation_error = ledger_full_error
                else:
                    reconciliation_summary, reconciliation_error = _call_reconciliation_agent(
                        bank_full, ledger_full
                    )
                    if reconciliation_summary:
                        try:
                            blocks = []
                            current = []
                            for raw_line in reconciliation_summary.splitlines():
                                if raw_line.strip() == "":
                                    if current:
                                        blocks.append(current)
                                        current = []
                                    continue
                                current.append(raw_line)
                            if current:
                                blocks.append(current)

                            unmatched_blocks = []
                            for block in blocks:
                                joined = "\n".join(block)
                                if "classification: unmatched" in joined.lower():
                                    unmatched_blocks.append(joined)

                            if unmatched_blocks:
                                unmatched_summary = "\n\n".join(unmatched_blocks)
                        except Exception:
                            unmatched_summary = None
        except Exception:
            reconciliation_error = "Unexpected error during reconciliation."

    return render_template(
        "index.html",
        bank_preview=bank_preview,
        ledger_preview=ledger_preview,
        bank_error=bank_error,
        ledger_error=ledger_error,
        reconciliation_summary=reconciliation_summary,
        reconciliation_error=reconciliation_error,
        unmatched_summary=unmatched_summary,
        password_error=password_error,
    )


if __name__ == "__main__":
    app.run(debug=False)
