"""
SIEM Client — Elasticsearch + Webhook integration
"""

import os, json, logging
from datetime import datetime

import requests

logger = logging.getLogger("ai-nids.siem")

ES_HOST          = os.getenv("ES_HOST", "localhost")
ES_PORT          = os.getenv("ES_PORT", "9200")
ES_INDEX         = os.getenv("ES_INDEX", "ai-nids-alerts")
SLACK_WEBHOOK    = os.getenv("SLACK_WEBHOOK_URL", "")
DRY_RUN          = os.getenv("SIEM_DRY_RUN", "true").lower() == "true"


def send_to_elasticsearch(event: dict) -> bool:
    """Forward event to Elasticsearch. Returns True on success."""
    event["@timestamp"] = datetime.utcnow().isoformat() + "Z"
    event["source"] = "AI-NIDS"

    if DRY_RUN:
        logger.debug(f"[DRY_RUN] ES: {json.dumps(event)}")
        return True

    try:
        url  = f"http://{ES_HOST}:{ES_PORT}/{ES_INDEX}/_doc"
        resp = requests.post(url, json=event, timeout=2)
        if resp.status_code not in (200, 201):
            logger.error(f"ES error {resp.status_code}: {resp.text}")
            return False
        return True
    except Exception as e:
        logger.error(f"ES connection error: {e}")
        return False


def trigger_webhook(severity: str, event: dict) -> bool:
    """Send Slack/PagerDuty webhook for HIGH/CRITICAL events."""
    color_map = {"CRITICAL": "danger", "HIGH": "warning", "MEDIUM": "good"}

    payload = {
        "text": f":rotating_light: *[{severity}] AI-NIDS Intrusion Alert*",
        "attachments": [{
            "color": color_map.get(severity, "good"),
            "fields": [
                {"title": "Attack Type", "value": event.get("label", "?"), "short": True},
                {"title": "Source IP",   "value": event.get("src_ip", "?"), "short": True},
                {"title": "AE Score",    "value": str(event.get("ae_score", 0)), "short": True},
                {"title": "Confidence",  "value": str(event.get("confidence", 0)), "short": True},
                {"title": "Timestamp",   "value": event.get("@timestamp", ""), "short": False},
            ],
        }],
    }

    if DRY_RUN:
        logger.info(f"[DRY_RUN] Webhook [{severity}]: {event.get('label')} from {event.get('src_ip')}")
        return True

    if not SLACK_WEBHOOK:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping webhook")
        return False

    try:
        resp = requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return False
