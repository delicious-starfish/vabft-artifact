#!/usr/bin/env python3
"""Send an email to the project notification address when this script runs.

The recipient is deliberately fixed.  Configure the sender outside the
repository with SMTP_USERNAME and SMTP_PASSWORD (for QQ Mail, the latter is
the SMTP authorization code, not the web-login password).
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage


RECIPIENT = "2036465770@qq.com"
DEFAULT_SMTP_HOST = "smtp.qq.com"
DEFAULT_SMTP_PORT = 465


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="V-ABFT notification")
    parser.add_argument("--body", default="The notification script was activated.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    username = required_env("SMTP_USERNAME")
    password = required_env("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM", username)
    smtp_host = os.environ.get("SMTP_HOST", DEFAULT_SMTP_HOST)
    smtp_port = int(os.environ.get("SMTP_PORT", str(DEFAULT_SMTP_PORT)))

    message = EmailMessage()
    message["From"] = sender
    message["To"] = RECIPIENT
    message["Subject"] = args.subject
    message.set_content(args.body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
        server.login(username, password)
        server.send_message(message)

    print(f"Email sent to {RECIPIENT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, smtplib.SMTPException, ValueError) as exc:
        print(f"Email was not sent: {exc}", file=sys.stderr)
        raise SystemExit(1)
