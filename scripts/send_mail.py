"""Send a plain-text email to the project owner via Gmail SMTP.

Credentials live in /home/go_ai/.secrets/gmail.env (chmod 600).

Usage:
    python scripts/send_mail.py --subject "..." --body-file report.txt
    echo "body" | python scripts/send_mail.py --subject "..."
"""
from __future__ import annotations

import argparse
import smtplib
import ssl
import sys
from email.mime.text import MIMEText
from pathlib import Path

SECRETS = Path("/home/go_ai/.secrets/gmail.env")


def load_creds():
    creds = {}
    for line in SECRETS.read_text().splitlines():
        line = line.strip()
        if line and "=" in line:
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()
    return creds["GMAIL_USER"], creds["GMAIL_APP_PASSWORD"]


def send(subject: str, body: str):
    user, password = load_creds()
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = user
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as s:
        s.login(user, password)
        s.sendmail(user, [user], msg.as_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--body-file", default=None)
    args = p.parse_args()
    body = Path(args.body_file).read_text() if args.body_file else sys.stdin.read()
    send(args.subject, body)
    print("sent")
