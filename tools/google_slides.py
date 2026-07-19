"""
google_slides.py
================
Read and edit a Google Slides presentation via the Slides API.

Setup (one-time): credentials.json in the project root (OAuth Desktop client).
First run opens a browser to authorize; token.json is cached afterwards.

Usage:
    # List every slide and text element (IDs + current text)
    python tools/google_slides.py --list PRESENTATION_ID

    # Replace the entire text of one element (keeps the element's base style)
    python tools/google_slides.py --set PRESENTATION_ID --element ELEMENT_ID --text "New text"

    # Find/replace a string everywhere in the deck
    python tools/google_slides.py --swap PRESENTATION_ID --find "old" --with "new"

The presentation ID is the long string in the deck URL:
    docs.google.com/presentation/d/1b5VFuZGyX1178noprYrdLpIFk6j98pZZ/edit
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/presentations"]
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(ROOT, "credentials.json")
TOKEN_FILE       = os.path.join(ROOT, "token.json")


def get_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # Test-mode refresh tokens expire after ~7 days — re-authorize
                creds = None
        if not creds or not creds.valid:
            if not os.path.exists(CREDENTIALS_FILE):
                sys.exit(f"credentials.json not found at {CREDENTIALS_FILE}")
            print("Google token expired — a browser window will open to re-authorize.")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("slides", "v1", credentials=creds)


def element_text(element):
    """Extract plain text from a page element, or None if it has no text."""
    shape = element.get("shape")
    if not shape or "text" not in shape:
        return None
    parts = []
    for te in shape["text"].get("textElements", []):
        run = te.get("textRun")
        if run:
            parts.append(run.get("content", ""))
    return "".join(parts)


def _print_element(el, indent="  "):
    kind = ("shape" if "shape" in el else
            "image" if "image" in el else
            "table" if "table" in el else
            "group" if "elementGroup" in el else "other")
    txt = element_text(el)
    if txt is not None:
        preview = txt.strip().replace("\n", " / ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"{indent}[{el['objectId']}] {kind}: {preview}")
    else:
        print(f"{indent}[{el['objectId']}] {kind}")
    # Recurse into groups so grouped text is visible
    for child in el.get("elementGroup", {}).get("children", []):
        _print_element(child, indent + "    ")


def cmd_list(service, pres_id):
    pres = service.presentations().get(presentationId=pres_id).execute()
    print(f"Title: {pres.get('title')}")
    print(f"Slides: {len(pres.get('slides', []))}\n")
    for i, slide in enumerate(pres.get("slides", []), 1):
        print(f"=== Slide {i}  (id: {slide['objectId']}) ===")
        for el in slide.get("pageElements", []):
            _print_element(el)
        print()


def cmd_set(service, pres_id, element_id, text):
    requests = [
        {"deleteText": {"objectId": element_id, "textRange": {"type": "ALL"}}},
        {"insertText": {"objectId": element_id, "insertionIndex": 0, "text": text}},
    ]
    service.presentations().batchUpdate(
        presentationId=pres_id, body={"requests": requests}).execute()
    print(f"Replaced text of element {element_id} ({len(text)} chars).")


def cmd_swap(service, pres_id, find, replace):
    requests = [{
        "replaceAllText": {
            "containsText": {"text": find, "matchCase": True},
            "replaceText": replace,
        }
    }]
    resp = service.presentations().batchUpdate(
        presentationId=pres_id, body={"requests": requests}).execute()
    n = resp["replies"][0].get("replaceAllText", {}).get("occurrencesChanged", 0)
    print(f"Replaced {n} occurrence(s) of {find!r}.")


def main():
    p = argparse.ArgumentParser(description="Read/edit a Google Slides deck.")
    p.add_argument("--list", metavar="PRES_ID", help="List slides and text elements")
    p.add_argument("--set",  metavar="PRES_ID", help="Replace one element's text")
    p.add_argument("--element", help="Element ID (with --set)")
    p.add_argument("--text",    help="New text (with --set)")
    p.add_argument("--swap", metavar="PRES_ID", help="Find/replace across the deck")
    p.add_argument("--find",             help="Text to find (with --swap)")
    p.add_argument("--with", dest="repl", help="Replacement text (with --swap)")
    args = p.parse_args()

    service = get_service()

    if args.list:
        cmd_list(service, args.list)
    elif args.set:
        if not args.element or args.text is None:
            sys.exit("--set requires --element and --text")
        cmd_set(service, args.set, args.element, args.text)
    elif args.swap:
        if not args.find or args.repl is None:
            sys.exit("--swap requires --find and --with")
        cmd_swap(service, args.swap, args.find, args.repl)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
